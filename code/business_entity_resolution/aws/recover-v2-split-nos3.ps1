[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 5)]
    [int]$Shard
)

$ErrorActionPreference = 'Stop'
$env:AWS_SDK_UA_APP_ID = 'AWSSkill-EC2'

$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$metadataPath = Join-Path $workspace 'output\v2\latest-split-run.json'
$run = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
$worker = @($run.workers | Where-Object { [int]$_.shard_index -eq $Shard })[0]
if (-not $worker) { throw "Shard $Shard is not present in the latest split run." }

$aws = Get-Command aws.exe -ErrorAction Stop
$sshKeygen = Get-Command ssh-keygen.exe -ErrorAction Stop
$scp = Get-Command scp.exe -ErrorAction Stop

function Invoke-QuietProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string]$Arguments
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = $Arguments
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    [void]$process.Start()
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    [pscustomobject]@{
        ExitCode = $process.ExitCode
        Output = (($stdoutTask.Result + $stderrTask.Result).Trim())
    }
}

function Copy-RemoteFile {
    param([Parameter(Mandatory = $true)][string]$RemotePath)

    $push = @(
        'ec2-instance-connect', 'send-ssh-public-key', '--region', [string]$worker.region,
        '--instance-id', [string]$worker.instance_id, '--instance-os-user', 'ec2-user',
        '--availability-zone', [string]$instance.az, '--ssh-public-key', "file://$publicKey",
        '--output', 'json'
    )
    $pushText = (& $aws.Source @push 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        return [pscustomobject]@{ ExitCode = 255; Output = "EC2 Instance Connect failed: $pushText" }
    }

    $scpArgs = '-i "{0}" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL -o GlobalKnownHostsFile=NUL -o ConnectTimeout=20 "{1}" "{2}"' -f $privateKey, $RemotePath, $destination
    Invoke-QuietProcess -FilePath $scp.Source -Arguments $scpArgs
}

$describe = @(
    'ec2', 'describe-instances', '--region', [string]$worker.region,
    '--instance-ids', [string]$worker.instance_id,
    '--query', 'Reservations[0].Instances[0].{state:State.Name,az:Placement.AvailabilityZone,ip:PublicIpAddress}',
    '--output', 'json'
)
$instanceText = (& $aws.Source @describe 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw "EC2 lookup failed: $instanceText" }
$instance = $instanceText | ConvertFrom-Json
if ($instance.state -ne 'running') {
    throw "Shard $Shard EC2 state is $($instance.state); direct SSH recovery requires it to be running."
}

$destination = Join-Path $workspace ('output\v2-recovered\live-snapshot\shard-{0:D3}' -f $Shard)
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$keyDirectory = Join-Path ([IO.Path]::GetTempPath()) ('devcore-copy-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $keyDirectory | Out-Null
$privateKey = Join-Path $keyDirectory 'copy-key'
$publicKey = "$privateKey.pub"
$knownHosts = Join-Path $keyDirectory 'known_hosts'
New-Item -ItemType File -Path $knownHosts | Out-Null

try {
    $keygenArgs = '-q -t rsa -b 2048 -f "{0}" -N ""' -f $privateKey
    $keygenResult = Invoke-QuietProcess -FilePath $sshKeygen.Source -Arguments $keygenArgs
    if ($keygenResult.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $publicKey)) {
        throw "Could not create a temporary SSH key: $($keygenResult.Output)"
    }

    $shardName = '{0:D3}' -f $Shard
    $suffix = 'shard-{0}-of-{1:D3}' -f $shardName, [int]$run.shard_count
    $transferErrors = New-Object System.Collections.Generic.List[string]
    $logSource = 'ec2-user@{0}:/var/log/devcore-v2-split.log' -f $instance.ip
    $artifactNames = [System.Collections.Generic.List[string]]::new()
    [void]$artifactNames.Add('feature-shard-metrics.json')
    [void]$artifactNames.Add("train_features-$suffix.parquet")
    [void]$artifactNames.Add("train_targets-$suffix.parquet")

    $snapshotNote = @(
        'LIVE SNAPSHOT: copied while the worker may still be writing.',
        'Check Parquet integrity and recopy after shard completion before using as final data.',
        "Run: $($run.run_id)",
        "Shard: $Shard",
        "Instance: $($worker.instance_id)",
        "Captured: $((Get-Date).ToString('o'))"
    )
    $snapshotNote | Set-Content -LiteralPath (Join-Path $destination 'SNAPSHOT_STATUS.txt')

    Write-Host "Copying shard $Shard directly from $($instance.ip) to $destination (no S3)."
    Write-Host "  Fetching $logSource"
    $logResult = Copy-RemoteFile -RemotePath $logSource
    if ($logResult.ExitCode -ne 0) {
        $transferErrors.Add("SCP failed for $logSource`: $($logResult.Output)")
    }

    foreach ($artifactName in $artifactNames) {
        $remoteCandidates = [System.Collections.Generic.List[string]]::new()
        [void]$remoteCandidates.Add(('ec2-user@{0}:/opt/devcore-v2/results/feature-shards/{1}/{2}' -f $instance.ip, $shardName, $artifactName))
        [void]$remoteCandidates.Add(('ec2-user@{0}:/opt/devcore-v2/results/{1}' -f $instance.ip, $artifactName))
        $copied = $false
        foreach ($remoteFile in $remoteCandidates) {
            Write-Host "  Fetching $remoteFile"
            $copyResult = Copy-RemoteFile -RemotePath $remoteFile
            if ($copyResult.ExitCode -eq 0) { $copied = $true; break }
        }
        if (-not $copied) {
            $message = "No copyable $artifactName found in either worker output location."
            $transferErrors.Add($message)
            Write-Warning $message
        }
    }

    Get-ChildItem -LiteralPath $destination -File |
        Select-Object Name, Length |
        Format-Table -AutoSize
    if ($transferErrors.Count -gt 0) {
        $transferErrors | Set-Content -LiteralPath (Join-Path $destination 'TRANSFER_ERRORS.txt')
        Write-Warning "Shard $Shard copied what was available; $($transferErrors.Count) file(s) were missing or failed."
    }
}
finally {
    if (Test-Path -LiteralPath $keyDirectory) {
        Remove-Item -LiteralPath $keyDirectory -Recurse -Force
    }
}
