[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ResultsDirectory
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$results = (Resolve-Path -LiteralPath $ResultsDirectory).Path
$matching = Join-Path $results 'matching_results.tsv'
$candidate = Join-Path $results 'candidate_pairs.tsv'
$oofPath = Join-Path $results 'oof_metrics.json'
$testMetricsPath = Join-Path $results 'test_metrics.json'
foreach ($path in @($matching, $candidate, $oofPath, $testMetricsPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing submission file: $path" }
}
$oof = Get-Content -LiteralPath $oofPath -Raw | ConvertFrom-Json
$testMetrics = Get-Content -LiteralPath $testMetricsPath -Raw | ConvertFrom-Json
$operating = $oof.operating_point
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$number = { param($value, $format) ([double]$value).ToString($format, $culture) }
$distribution = 'mean {0}; median {1}; p95 {2}; p99 {3}; max {4}' -f @(
    (& $number $testMetrics.candidate_count_mean '0.###'),
    (& $number $testMetrics.candidate_count_median '0.###'),
    (& $number $testMetrics.candidate_count_p95 '0.###'),
    (& $number $testMetrics.candidate_count_p99 '0.###'),
    [int]$testMetrics.candidate_count_max
)
$replacements = @{
    '[[SELECTED_CANDIDATES_PER_TARGET]]' = [string]$operating.cap_per_target
    '[[TEST_CANDIDATE_PAIRS]]' = [string]$testMetrics.candidate_pairs
    '[[CANDIDATE_DISTRIBUTION]]' = $distribution
    '[[CANDIDATE_REDUCTION]]' = (& $number $testMetrics.candidate_reduction_ratio 'P2')
    '[[SELECTED_CAP]]' = [string]$operating.cap_per_target
    '[[SELECTED_THRESHOLD]]' = (& $number $operating.threshold '0.####')
    '[[CALIBRATION_F05]]' = (& $number $operating.macro_f0_5 '0.000000')
    '[[AUDIT_F05]]' = (& $number $operating.audit_macro_f0_5 '0.000000')
}

$studentRoot = Join-Path $workspace '6ab10eb3b23ba_student_resource\student_resource'
$validator = Join-Path $studentRoot 'utils\validate_submission.py'
$testDirectory = Join-Path $studentRoot 'dataset\test'
Push-Location $studentRoot
try {
    & python $validator --matching $matching --candidate $candidate --test-dir $testDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Official submission validation failed; current output files were kept.' }
}
finally {
    Pop-Location
}

$output = Join-Path $workspace 'output'
$stamp = [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss')
$backup = Join-Path $output "baseline-backup-$stamp"
New-Item -ItemType Directory -Force -Path $backup | Out-Null
foreach ($name in @('matching_results.tsv', 'candidate_pairs.tsv')) {
    $current = Join-Path $output $name
    if (Test-Path -LiteralPath $current -PathType Leaf) {
        Copy-Item -LiteralPath $current -Destination (Join-Path $backup $name)
    }
}
$documentation = Join-Path $workspace 'Documentation_template.md'
if (Test-Path -LiteralPath $documentation -PathType Leaf) {
    Copy-Item -LiteralPath $documentation -Destination (Join-Path $backup 'Documentation_template.md')
}
$archive = Join-Path $workspace 'DevCore_submission.zip'
if (Test-Path -LiteralPath $archive -PathType Leaf) {
    Copy-Item -LiteralPath $archive -Destination (Join-Path $backup 'DevCore_submission.zip')
}

Copy-Item -LiteralPath $matching -Destination (Join-Path $output 'matching_results.tsv') -Force
Copy-Item -LiteralPath $candidate -Destination (Join-Path $output 'candidate_pairs.tsv') -Force
$methodology = Get-Content -LiteralPath $documentation -Raw
foreach ($token in $replacements.Keys) { $methodology = $methodology.Replace($token, [string]$replacements[$token]) }
if ($methodology -match '\[\[[A-Z0-9_]+\]\]') { throw 'Methodology metrics were not fully populated.' }
[IO.File]::WriteAllText($documentation, $methodology, [Text.UTF8Encoding]::new($false))
$promotionStarted = [DateTime]::UtcNow
& (Join-Path $PSScriptRoot 'package-submission.ps1')
if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) { throw 'Submission archive was not created.' }
$archiveInfo = Get-Item -LiteralPath $archive
if ($archiveInfo.LastWriteTimeUtc -lt $promotionStarted) { throw 'The submission archive was not refreshed.' }
Write-Output "Validated and packaged the v2 results. Previous files are saved in: $backup"
