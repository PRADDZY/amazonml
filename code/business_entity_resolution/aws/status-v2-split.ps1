[CmdletBinding()]
param(
    [string]$RunId,
    [ValidateSet('features','fit','test','finalize')][string]$Phase
)

$ErrorActionPreference = 'Stop'
$env:AWS_SDK_UA_APP_ID = 'AWSSkill-EC2'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$latestPath = Join-Path $workspace 'output\v2\latest-split-run.json'
if (-not $RunId) {
    if (-not (Test-Path -LiteralPath $latestPath)) { throw 'No split run is recorded.' }
    $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
    $RunId = $latest.run_id
    if (-not $Phase) { $Phase = $latest.last_phase }
}
elseif (-not $Phase -and (Test-Path -LiteralPath $latestPath)) {
    $latest = Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json
    if ($latest.run_id -eq $RunId) { $Phase = $latest.last_phase }
}
if (-not $Phase) { throw 'Pass -Phase when checking a run that is not the latest.' }

$account = (aws sts get-caller-identity --region us-east-1 --query Account --output text).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Could not read the AWS account identity.' }
$bucket = "amazonml-er-$account-us-east-1-20260926"
$prefix = "output/v2/split-$RunId"
$manifestPath = Join-Path $workspace "output\v2\split-$RunId\$Phase-workers.json"
if (-not (Test-Path -LiteralPath $manifestPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $manifestPath) | Out-Null
    aws s3 cp "s3://$bucket/$prefix/control/$Phase-workers.json" $manifestPath --region us-east-1 --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw "No worker manifest found for $RunId/$Phase." }
}
$workers = @(Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json)
$rows = foreach ($worker in $workers) {
    $status = 'NOT_REPORTED'
    $value = aws s3 cp $worker.status_uri - --region us-east-1 --only-show-errors 2>$null
    if ($LASTEXITCODE -eq 0) { $status = ($value | Out-String).Trim() }
    [pscustomobject]@{
        shard = $worker.shard_index
        region = $worker.region
        instance = $worker.instance_id
        status = $status
    }
}
$rows | Sort-Object shard | Format-Table -AutoSize
$succeeded = @($rows | Where-Object status -eq 'SUCCEEDED').Count
$failed = @($rows | Where-Object { $_.status -like 'FAILED*' }).Count
Write-Output "$succeeded/$($rows.Count) succeeded; $failed failed. Run $RunId, phase $Phase."
if ($succeeded -eq $rows.Count -and $rows.Count -gt 0) {
    $next = switch ($Phase) {
        'features' { 'fit' }
        'fit' { 'test' }
        'test' { 'finalize' }
        'finalize' { 'fetch and validate the final files' }
    }
    Write-Output "Next: $next"
}
