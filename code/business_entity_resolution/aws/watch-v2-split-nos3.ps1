[CmdletBinding()]
param(
    [ValidateRange(60, 3600)]
    [int]$IntervalSeconds = 300,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$env:AWS_SDK_UA_APP_ID = 'AWSSkill-EC2'

$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$outputRoot = Join-Path $workspace 'output\v2'
$latestPath = Join-Path $outputRoot 'latest-split-run.json'
if (-not (Test-Path -LiteralPath $latestPath)) {
    throw "Split run metadata not found: $latestPath"
}
$run = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
$phaseName = ([string]$run.last_phase).ToUpperInvariant()
$workers = @($run.workers | Sort-Object -Property shard_index)
if ($workers.Count -eq 0) { throw 'The latest split run has no worker records.' }

$ssh = Get-Command ssh.exe -ErrorAction Stop
$sshKeygen = Get-Command ssh-keygen.exe -ErrorAction Stop
$keyDirectory = Join-Path ([IO.Path]::GetTempPath()) ('devcore-status-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $keyDirectory | Out-Null
$privateKeyPath = Join-Path $keyDirectory 'status-key'
$publicKeyPath = "$privateKeyPath.pub"
$knownHostsPath = Join-Path $keyDirectory 'known_hosts'
New-Item -ItemType File -Path $knownHostsPath | Out-Null

& $sshKeygen.Source -q -t rsa -b 2048 -f $privateKeyPath -N '""' 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $publicKeyPath)) {
    throw 'Could not create a temporary SSH key for EC2 Instance Connect.'
}

function Invoke-AwsCapture {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $captured = (& aws @Arguments 2>&1 | Out-String).Trim()
        $code = $LASTEXITCODE
    }
    catch {
        $captured = $_.Exception.Message
        $code = 127
    }
    finally {
        $ErrorActionPreference = $savedPreference
    }
    [pscustomobject]@{ ExitCode = [int]$code; Text = [string]$captured }
}

function Get-RemoteSnapshot {
    param(
        [Parameter(Mandatory = $true)]$Worker,
        [Parameter(Mandatory = $true)]$Instance
    )

    $push = Invoke-AwsCapture -Arguments @(
        'ec2-instance-connect', 'send-ssh-public-key', '--region', [string]$Worker.region,
        '--instance-id', [string]$Worker.instance_id, '--instance-os-user', 'ec2-user',
        '--availability-zone', [string]$Instance.Az, '--ssh-public-key', "file://$publicKeyPath", '--output', 'json'
    )
    if ($push.ExitCode -ne 0) {
        return [pscustomobject]@{ Reachable = $false; Error = "EC2 Instance Connect: $($push.Text)"; Text = '' }
    }

    $remoteScript = @'
set +e
stage=$(cat /tmp/devcore-v2-split-status.txt 2>/dev/null | head -n 1)
[ -z "$stage" ] && stage=NO_LOCAL_STATUS
printf 'STAGE=%s\n' "$stage"
if pgrep -f '[v]2_pipeline.py' >/dev/null; then process=active; else process=none; fi
printf 'PROCESS=%s\n' "$process"
checkpoint=$(grep -E 'Built features for|Cross-fit fold|Selected operating point|Training feature collection complete|STEP_[A-Z_]+|Traceback|RuntimeError|FAILED|SUCCEEDED|Killed|candidate_pairs|test targets|candidate pairs' /var/log/devcore-v2-split.log 2>/dev/null | tail -n 1)
[ -z "$checkpoint" ] && checkpoint=none
printf 'CHECKPOINT=%s\n' "$checkpoint"
files=$(find /opt/devcore-v2/results -maxdepth 1 -type f -printf '%f=%sB ' 2>/dev/null)
[ -z "$files" ] && files=none
printf 'FILES=%s\n' "$files"
disk=$(df -h / | awk 'NR==2 {print $3 "/" $2 " used"}')
printf 'DISK=%s\n' "$disk"
'@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($remoteScript))
    $remoteCommand = "echo $encoded | base64 -d | bash"
    $sshArguments = @(
        '-i', $privateKeyPath,
        '-o', 'BatchMode=yes',
        '-o', 'IdentitiesOnly=yes',
        '-o', 'StrictHostKeyChecking=no',
        '-o', "UserKnownHostsFile=$knownHostsPath",
        '-o', 'GlobalKnownHostsFile=NUL',
        '-o', 'ConnectTimeout=15',
        "ec2-user@$($Instance.PublicIp)",
        $remoteCommand
    )

    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $text = (& $ssh.Source @sshArguments 2>&1 | Out-String).Trim()
        $code = $LASTEXITCODE
    }
    catch {
        $text = $_.Exception.Message
        $code = 127
    }
    finally {
        $ErrorActionPreference = $savedPreference
    }
    if ($code -ne 0) {
        return [pscustomobject]@{ Reachable = $false; Error = $text; Text = $text }
    }
    [pscustomobject]@{ Reachable = $true; Error = ''; Text = $text }
}

function Invoke-StatusCheck {
    $checked = Get-Date
    Write-Host ''
    Write-Host ('=' * 100)
    Write-Host 'DEVCORE - AWS ENTITY MATCHING (S3-FREE WORKER CHECK)'
    Write-Host "Run: $($run.run_id) | Phase: $phaseName | Checked: $($checked.ToString('yyyy-MM-dd HH:mm:ss')) local time"

    try {
        $publicIp = (Invoke-RestMethod -Uri 'https://checkip.amazonaws.com' -TimeoutSec 15).Trim()
    }
    catch {
        Write-Host "Cannot determine this computer's public IPv4 address: $($_.Exception.Message)"
        return
    }
    if ($publicIp -notmatch '^\d{1,3}(\.\d{1,3}){3}$') {
        Write-Host "Unexpected public IPv4 response: $publicIp"
        return
    }
    $cidr = "$publicIp/32"
    Write-Host "Access path: EC2 Instance Connect over SSH, temporary inbound source $cidr"
    Write-Host 'Loading worker metadata from AWS EC2...'

    $instanceInfo = @{}
    foreach ($region in @($workers | Select-Object -ExpandProperty region -Unique)) {
        $regionWorkers = @($workers | Where-Object { [string]$_.region -eq [string]$region })
        Write-Host "  $region`: looking up $($regionWorkers.Count) worker(s)"
        $describeArguments = @('ec2', 'describe-instances', '--region', [string]$region, '--instance-ids')
        $describeArguments += @($regionWorkers | ForEach-Object { [string]$_.instance_id })
        $describeArguments += @(
            '--query', 'Reservations[].Instances[].{InstanceId:InstanceId,State:State.Name,Az:Placement.AvailabilityZone,PublicIp:PublicIpAddress,GroupId:NetworkInterfaces[0].Groups[0].GroupId}',
            '--output', 'json'
        )
        $instancesResult = Invoke-AwsCapture -Arguments $describeArguments
        if ($instancesResult.ExitCode -ne 0) {
            Write-Host "EC2 lookup failed in $region`: $($instancesResult.Text)"
            continue
        }
        # In Windows PowerShell 5.1, piping a JSON array through ConvertFrom-Json
        # can keep the whole array as one nested object. Parse via -InputObject so
        # multi-instance regions produce one record per instance.
        $parsedInstances = ConvertFrom-Json -InputObject $instancesResult.Text
        foreach ($instance in $parsedInstances) {
            $instanceInfo[[string]$instance.InstanceId] = $instance
        }
        Write-Host "  $region`: found $(@($parsedInstances).Count) worker(s)"
    }

    $createdRules = New-Object System.Collections.Generic.List[object]
    try {
        $groups = @{}
        foreach ($worker in $workers) {
            $instance = $instanceInfo[[string]$worker.instance_id]
            if ($instance -and [string]$instance.State -eq 'running' -and $instance.PublicIp -and $instance.GroupId) {
                $groupKey = "$( [string]$worker.region )|$( [string]$instance.GroupId )"
                if (-not $groups.ContainsKey($groupKey)) {
                    $groups[$groupKey] = [pscustomobject]@{ Region = [string]$worker.region; GroupId = [string]$instance.GroupId }
                }
            }
        }
        foreach ($group in $groups.Values) {
            Write-Host "Opening temporary SSH access in $($group.Region)..."
            $authorize = Invoke-AwsCapture -Arguments @(
                'ec2', 'authorize-security-group-ingress', '--region', $group.Region, '--group-id', $group.GroupId,
                '--protocol', 'tcp', '--port', '22', '--cidr', $cidr,
                '--tag-specifications', 'ResourceType=security-group-rule,Tags=[{Key=Purpose,Value=DevCoreTemporaryStatusPoll}]',
                '--output', 'json'
            )
            if ($authorize.ExitCode -eq 0) {
                $created = $authorize.Text | ConvertFrom-Json
                foreach ($rule in @($created.SecurityGroupRules)) {
                    $createdRules.Add([pscustomobject]@{ Region = $group.Region; GroupId = $group.GroupId; RuleId = $rule.SecurityGroupRuleId })
                }
            }
            elseif ($authorize.Text -match 'InvalidPermission\.Duplicate') {
                Write-Host "A matching temporary SSH rule already exists for $($group.Region) / $($group.GroupId); leaving it unchanged."
            }
            else {
                Write-Host "Could not open temporary SSH access for $($group.Region): $($authorize.Text)"
            }
        }

        $rows = New-Object System.Collections.Generic.List[object]
        foreach ($worker in $workers) {
            $shard = [int]$worker.shard_index
            $region = [string]$worker.region
            $instanceId = [string]$worker.instance_id
            $instance = $instanceInfo[$instanceId]
            if (-not $instance) {
                $rows.Add([pscustomobject]@{ Shard = $shard; Region = $region; Instance = $instanceId; State = 'UNKNOWN'; Stage = 'EC2 lookup failed'; Progress = '' })
                continue
            }
            if ([string]$instance.State -ne 'running') {
                $rows.Add([pscustomobject]@{ Shard = $shard; Region = $region; Instance = $instanceId; State = [string]$instance.State; Stage = 'Not running'; Progress = 'Local EBS data remains attached while stopped.' })
                continue
            }
            if (-not $instance.PublicIp -or -not $instance.Az) {
                $rows.Add([pscustomobject]@{ Shard = $shard; Region = $region; Instance = $instanceId; State = 'running'; Stage = 'No public SSH route'; Progress = '' })
                continue
            }

            Write-Host "Reading shard $shard on $region ($instanceId)..."
            $snapshot = Get-RemoteSnapshot -Worker $worker -Instance $instance
            if (-not $snapshot.Reachable) {
                $rows.Add([pscustomobject]@{ Shard = $shard; Region = $region; Instance = $instanceId; State = 'running'; Stage = 'SSH unavailable'; Progress = $snapshot.Error })
                continue
            }
            $stageMatch = [regex]::Match($snapshot.Text, '(?m)^STAGE=(.*)$')
            $progressMatch = [regex]::Match($snapshot.Text, '(?m)^CHECKPOINT=(.*)$')
            $processMatch = [regex]::Match($snapshot.Text, '(?m)^PROCESS=(.*)$')
            $filesMatch = [regex]::Match($snapshot.Text, '(?m)^FILES=(.*)$')
            $diskMatch = [regex]::Match($snapshot.Text, '(?m)^DISK=(.*)$')
            $stage = if ($stageMatch.Success) { $stageMatch.Groups[1].Value.Trim() } else { 'No local status marker' }
            $progress = if ($progressMatch.Success) { $progressMatch.Groups[1].Value.Trim() } else { 'No checkpoint found in local log.' }
            $process = if ($processMatch.Success) { $processMatch.Groups[1].Value.Trim() } else { 'unknown' }
            $files = if ($filesMatch.Success) { $filesMatch.Groups[1].Value.Trim() } else { 'unknown' }
            $disk = if ($diskMatch.Success) { $diskMatch.Groups[1].Value.Trim() } else { 'unknown' }
            $summary = "Process: $process | $progress | Files: $files | Disk: $disk"
            $rows.Add([pscustomobject]@{ Shard = $shard; Region = $region; Instance = $instanceId; State = 'running'; Stage = $stage; Progress = $summary })
        }

        Write-Host ''
        Write-Host 'WORKER DETAILS'
        foreach ($row in $rows) {
            Write-Host "Shard $($row.Shard) | $($row.Region) | EC2 $($row.State) | $($row.Stage)"
            if ($row.Progress) {
                foreach ($detail in ($row.Progress -split ' \| ')) {
                    Write-Host "  $detail"
                }
            }
        }
        $running = @($rows | Where-Object { $_.State -eq 'running' }).Count
        $reachable = @($rows | Where-Object { $_.Stage -ne 'SSH unavailable' -and $_.Stage -ne 'No public SSH route' -and $_.Stage -ne 'EC2 lookup failed' }).Count
        Write-Host ''
        Write-Host "Workers: $running/$($rows.Count) EC2 running | $reachable/$($rows.Count) local snapshots read"
    }
    finally {
        foreach ($rule in $createdRules) {
            $revoke = Invoke-AwsCapture -Arguments @(
                'ec2', 'revoke-security-group-ingress', '--region', $rule.Region, '--group-id', $rule.GroupId,
                '--security-group-rule-ids', $rule.RuleId, '--output', 'json'
            )
            if ($revoke.ExitCode -ne 0) {
                Write-Host "WARNING: Could not remove temporary rule $($rule.RuleId): $($revoke.Text)"
            }
        }
    }
}

try {
    $nextCheck = Get-Date
    do {
        $checkStarted = Get-Date
        try {
            Invoke-StatusCheck
        }
        catch {
            Write-Host "CHECK FAILED: $($_.Exception.Message)"
        }
        if ($Once) { break }
        $nextCheck = $checkStarted.AddSeconds($IntervalSeconds)
        $waitSeconds = [int][Math]::Ceiling(($nextCheck - (Get-Date)).TotalSeconds)
        Write-Host ''
        if ($waitSeconds -gt 0) {
            Write-Host "Next check at $($nextCheck.ToString('HH:mm:ss')) (every $([int]($IntervalSeconds / 60)) minutes). Press Ctrl+C to stop."
            Start-Sleep -Seconds $waitSeconds
        }
        else {
            Write-Host 'This check took longer than the interval; starting the next check now. Press Ctrl+C to stop.'
        }
    } while ($true)
}
finally {
    $temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    $fullKeyDirectory = [IO.Path]::GetFullPath($keyDirectory)
    if ($fullKeyDirectory.StartsWith($temporaryRoot, [StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $fullKeyDirectory).StartsWith('devcore-status-', [StringComparison]::OrdinalIgnoreCase)) {
        foreach ($path in @($publicKeyPath, $privateKeyPath, $knownHostsPath)) {
            if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
        }
        if (Test-Path -LiteralPath $fullKeyDirectory) { Remove-Item -LiteralPath $fullKeyDirectory -Force }
    }
}
