[CmdletBinding()]
param(
    [string]$RunId,
    [ValidateSet('features','fit','test','finalize')][string]$Phase
)

$ErrorActionPreference = 'Stop'
$env:AWS_SDK_UA_APP_ID = 'AWSSkill-EC2'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$outputRoot = Join-Path $workspace 'output\v2'
$latestPath = Join-Path $outputRoot 'latest-split-run.json'

function Invoke-AwsCapture {
    param([Parameter(Mandatory=$true)][string[]]$Arguments)

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
    [pscustomobject]@{ ExitCode=[int]$code; Text=[string]$captured }
}

function Limit-Text {
    param([string]$Value,[int]$Length = 130)
    $clean = ($Value -replace '[\r\n\t]+',' ').Trim()
    if ($clean.Length -gt $Length) { return $clean.Substring(0,$Length - 3) + '...' }
    return $clean
}

function Get-HumanStage {
    param([string]$Status)
    switch -Regex ($Status) {
        '^BOOTSTRAPPING$' { return 'Preparing worker and installing software' }
        '^DOWNLOADING_DATA$' { return 'Downloading training data and search indexes' }
        '^RUNNING_FEATURE_SHARD$' { return 'Generating training candidate features' }
        '^MERGING_FEATURES_AND_FITTING$' { return 'Training and calibrating the matching model' }
        '^RUNNING_TEST_SHARD$' { return 'Scoring test entities and building candidate pairs' }
        '^MERGING_TEST_SHARDS$' { return 'Combining shards and writing final TSV files' }
        '^UPLOADING_RESULTS$' { return 'Uploading results and logs to S3' }
        '^SUCCEEDED$' { return 'Finished successfully' }
        '^FAILED' { return 'Worker reported a failure' }
        '^NOT REPORTED$' { return 'No status file available yet' }
        default { return $Status }
    }
}

if (-not (Test-Path -LiteralPath $latestPath)) { throw "No split run metadata found at $latestPath" }
$latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
if (-not $RunId) { $RunId = [string]$latest.run_id }

$runMetadataPath = Join-Path $outputRoot "split-$RunId\run.json"
if ($RunId -eq [string]$latest.run_id) {
    $runMetadata = $latest
}
elseif (Test-Path -LiteralPath $runMetadataPath) {
    $runMetadata = Get-Content -LiteralPath $runMetadataPath -Raw | ConvertFrom-Json
}
else {
    throw "No local run metadata found for $RunId."
}
if (-not $Phase) {
    if ($RunId -eq [string]$latest.run_id) { $Phase = [string]$latest.last_phase }
    else { throw 'Pass -Phase when checking a run that is not the latest.' }
}
$runUri = [string]$runMetadata.run_uri
if (-not $runUri) { throw "Run metadata has no S3 URI for $RunId." }

$manifestPath = Join-Path $outputRoot "split-$RunId\$Phase-workers.json"
if (-not (Test-Path -LiteralPath $manifestPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $manifestPath) | Out-Null
    $manifestUri = "$runUri/control/$Phase-workers.json"
    $download = Invoke-AwsCapture -Arguments @('s3','cp',$manifestUri,$manifestPath,'--region','us-east-1','--only-show-errors')
    if ($download.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $manifestPath)) {
        throw "Could not download worker manifest: $(Limit-Text $download.Text 180)"
    }
}
$workers = @(Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json)
if ($workers.Count -eq 0) { throw "Worker manifest is empty: $manifestPath" }

$rows = @()
$issues = New-Object System.Collections.Generic.List[string]
foreach ($worker in ($workers | Sort-Object -Property shard_index)) {
    $regionName = [string]$worker.region
    $instanceId = [string]$worker.instance_id
    $ec2State = 'UNKNOWN'
    if ($regionName -and $instanceId) {
        $ec2Result = Invoke-AwsCapture -Arguments @('ec2','describe-instances','--region',$regionName,'--instance-ids',$instanceId,
            '--query','Reservations[0].Instances[0].State.Name','--output','text')
        if ($ec2Result.ExitCode -eq 0 -and $ec2Result.Text -and $ec2Result.Text -ne 'None') {
            $ec2State = $ec2Result.Text
        }
        elseif ($ec2Result.Text) {
            $issues.Add("Shard $($worker.shard_index) EC2 lookup: $(Limit-Text $ec2Result.Text 130)")
        }
    }

    $statusResult = Invoke-AwsCapture -Arguments @('s3','cp',[string]$worker.status_uri,'-','--region','us-east-1','--only-show-errors')
    if ($statusResult.ExitCode -eq 0 -and $statusResult.Text) {
        $status = Limit-Text $statusResult.Text 60
    }
    else {
        $status = 'NOT REPORTED'
        if ($statusResult.Text) { $issues.Add("Shard $($worker.shard_index) status: $(Limit-Text $statusResult.Text 130)") }
    }

    $progress = 'No worker log checkpoint available yet.'
    if ($status -notin @('SUCCEEDED') -and $status -notlike 'FAILED*') {
        $logUri = [string]$worker.status_uri -replace 'status\.txt$','job.log'
        $logResult = Invoke-AwsCapture -Arguments @('s3','cp',$logUri,'-','--region','us-east-1','--only-show-errors')
        if ($logResult.ExitCode -eq 0 -and $logResult.Text) {
            $logLines = @($logResult.Text -split '\r?\n' | Where-Object {
                $_ -match 'Built features for|Cross-fit fold|Selected operating point|Training feature collection complete|STEP_[A-Z_]+|Traceback|RuntimeError|FAILED|SUCCEEDED|Killed|Downloading|candidate_pairs|test targets'
            })
            if ($logLines.Count -gt 0) {
                $progress = Limit-Text $logLines[-1] 150
            }
            else {
                $lastLogLine = @($logResult.Text -split '\r?\n' | Where-Object { $_.Trim() } | Select-Object -Last 1)
                if ($lastLogLine.Count -gt 0) { $progress = Limit-Text $lastLogLine[0] 150 }
            }
        }
        elseif ($logResult.Text -match 'NoSuchKey|404|Not Found') {
            $progress = 'Worker log has not been uploaded yet.'
        }
        elseif ($logResult.ExitCode -ne 0) {
            $progress = "Log unavailable: $(Limit-Text $logResult.Text 100)"
        }
    }

    $rows += [pscustomobject]@{
        Shard = [int]$worker.shard_index
        Region = $regionName
        Instance = $instanceId
        Ec2State = $ec2State
        Status = $status
        Stage = Get-HumanStage $status
        Progress = $progress
    }
}

$succeeded = @($rows | Where-Object Status -eq 'SUCCEEDED').Count
$failed = @($rows | Where-Object { $_.Status -like 'FAILED*' }).Count
$unreported = @($rows | Where-Object Status -eq 'NOT REPORTED').Count
$working = $rows.Count - $succeeded - $failed - $unreported
$checked = Get-Date
$checkedUtc = $checked.ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss') + ' UTC'
$phaseLabel = $Phase.ToUpperInvariant()
$commonStages = @($rows | Where-Object { $_.Status -ne 'NOT REPORTED' } | Select-Object -ExpandProperty Stage -Unique)
if ($commonStages.Count -eq 1) { $summaryStage = $commonStages[0] }
elseif ($failed -gt 0) { $summaryStage = 'One or more workers reported a failure; inspect details below' }
elseif ($working -gt 0) { $summaryStage = 'Workers are at different steps in the pipeline' }
else { $summaryStage = 'Waiting for workers to report their current step' }

$display = New-Object System.Collections.Generic.List[string]
$display.Add('DEVCORE - AWS ENTITY MATCHING RUN')
$display.Add(('=' * 104))
$display.Add("Run:       $RunId")
$display.Add("Phase:     $phaseLabel")
$display.Add("Checked:   $($checked.ToString('yyyy-MM-dd HH:mm:ss')) IST  ($checkedUtc)")
$display.Add("Overall:   $succeeded/$($rows.Count) finished | $working in progress | $failed failed | $unreported not reported")
$display.Add("What it means: $summaryStage")
$display.Add('')
$display.Add(('{0,-6} {1,-13} {2,-21} {3,-11} {4,-25} {5}' -f 'SHARD','REGION','INSTANCE ID','EC2 STATE','WORKER STATUS','CURRENT STEP'))
$display.Add(('{0,-6} {1,-13} {2,-21} {3,-11} {4,-25} {5}' -f ('-'*5),('-'*6),('-'*11),('-'*9),('-'*13),('-'*12)))
foreach ($row in $rows) {
    $display.Add(('{0,-6} {1,-13} {2,-21} {3,-11} {4,-25} {5}' -f $row.Shard,$row.Region,$row.Instance,$row.Ec2State,$row.Status,$row.Stage))
}
$display.Add('')
$display.Add('LATEST WORKER LOG DETAIL')
foreach ($row in $rows) { $display.Add(("  Shard {0} ({1}): {2}" -f $row.Shard,$row.Region,$row.Progress)) }
if ($issues.Count -gt 0) {
    $display.Add('')
    $display.Add('AWS CHECK WARNINGS')
    foreach ($issue in $issues) { $display.Add("  - $issue") }
}
if ($succeeded -eq $rows.Count -and $failed -eq 0 -and $rows.Count -gt 0) {
    $next = switch ($Phase) {
        'features' { 'fit' }
        'fit' { 'test' }
        'test' { 'finalize' }
        'finalize' { 'fetch and validate the final files' }
    }
    $display.Add('')
    $display.Add("All workers finished successfully. Next: $next")
}
Write-Output ($display -join [Environment]::NewLine)
