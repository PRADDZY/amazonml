$ErrorActionPreference = 'Stop'
$env:AWS_SDK_UA_APP_ID = 'AWSSkill-SageMaker'
$region = 'us-east-1'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$runPath = Join-Path $workspace 'output\v2\latest-run.json'
if (-not (Test-Path -LiteralPath $runPath)) { throw "Missing run metadata: $runPath" }
$run = Get-Content -LiteralPath $runPath -Raw | ConvertFrom-Json
if ($run.mode -ne 'full') { throw "Latest run is '$($run.mode)'; only a full run contains submission outputs." }

$statusPath = Join-Path $workspace 'output\v2\latest-status.txt'
& aws s3 cp "$($run.run_uri)/status.txt" $statusPath --region $region --only-show-errors
if ($LASTEXITCODE -ne 0) { throw 'Could not read the full-run status from S3.' }
$status = (Get-Content -LiteralPath $statusPath -Raw).Trim()
if ($status -ne 'SUCCEEDED') { throw "Full run is not complete: $status" }

$stamp = ([DateTime]::Parse($run.started_utc).ToUniversalTime()).ToString('yyyyMMdd-HHmmss')
$destination = Join-Path $workspace "output\v2\results-$stamp"
New-Item -ItemType Directory -Force -Path $destination | Out-Null
& aws s3 sync "$($run.run_uri)/results/" $destination --region $region --only-show-errors
if ($LASTEXITCODE -ne 0) { throw 'Could not download the full-run results from S3.' }

foreach ($name in @('matching_results.tsv', 'candidate_pairs.tsv', 'metrics.json', 'test_metrics.json', 'oof_metrics.json')) {
    $path = Join-Path $destination $name
    if (-not (Test-Path -LiteralPath $path)) { throw "Full-run artifact is missing: $name" }
}
$run | Add-Member -NotePropertyName downloaded_utc -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
$run | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $destination 'run.json') -Encoding utf8
Write-Output "Downloaded full-run artifacts: $destination"
