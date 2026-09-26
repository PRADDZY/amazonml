[CmdletBinding()]
param([string]$RunId)

$ErrorActionPreference = 'Stop'
$env:AWS_SDK_UA_APP_ID = 'AWSSkill-EC2'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$latestPath = Join-Path $workspace 'output\v2\latest-split-run.json'
if (-not $RunId) {
    if (-not (Test-Path -LiteralPath $latestPath)) { throw 'No split run is recorded.' }
    $RunId = (Get-Content -LiteralPath $latestPath -Raw | ConvertFrom-Json).run_id
}
$account = (aws sts get-caller-identity --region us-east-1 --query Account --output text).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Could not read AWS account identity.' }
$bucket = "amazonml-er-$account-us-east-1-20260926"
$runUri = "s3://$bucket/output/v2/split-$RunId"
$manifestPath = Join-Path $workspace "output\v2\split-$RunId\finalize-workers.json"
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "Missing finalizer manifest: $manifestPath" }
$workers = @(Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json)
if ($workers.Count -ne 1) { throw 'Expected one finalizer worker.' }
$status = (aws s3 cp $workers[0].status_uri - --region us-east-1 --only-show-errors | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or $status -ne 'SUCCEEDED') { throw "Finalizer is not complete: $status" }

$destination = Join-Path $workspace "output\v2\results-$RunId"
New-Item -ItemType Directory -Force -Path $destination | Out-Null
aws s3 sync "$runUri/final/" $destination --region us-east-1 --only-show-errors
if ($LASTEXITCODE -ne 0) { throw 'Could not download the final sharded results.' }
foreach ($name in @('matching_results.tsv','candidate_pairs.tsv','metrics.json','test_metrics.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $destination $name) -PathType Leaf)) {
        throw "Final sharded artifact is missing: $name"
    }
}
foreach ($name in @('oof_metrics.json','fit_metrics.json')) {
    aws s3 cp "$runUri/fit/$name" (Join-Path $destination $name) --region us-east-1 --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw "Could not download $name from the fit phase." }
}
$run = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$run | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $destination 'run.json') -Encoding utf8
Write-Output "Downloaded split-run artifacts: $destination"
Write-Output "Validate and package with: .\code\business_entity_resolution\aws\promote-v2.ps1 -ResultsDirectory .\output\v2\results-$RunId"
