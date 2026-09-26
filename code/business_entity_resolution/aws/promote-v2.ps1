[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ResultsDirectory
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$results = (Resolve-Path -LiteralPath $ResultsDirectory).Path
$matching = Join-Path $results 'matching_results.tsv'
$candidate = Join-Path $results 'candidate_pairs.tsv'
foreach ($path in @($matching, $candidate)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing submission file: $path" }
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
$archive = Join-Path $workspace 'DevCore_submission.zip'
if (Test-Path -LiteralPath $archive -PathType Leaf) {
    Copy-Item -LiteralPath $archive -Destination (Join-Path $backup 'DevCore_submission.zip')
}

Copy-Item -LiteralPath $matching -Destination (Join-Path $output 'matching_results.tsv') -Force
Copy-Item -LiteralPath $candidate -Destination (Join-Path $output 'candidate_pairs.tsv') -Force
$promotionStarted = [DateTime]::UtcNow
& (Join-Path $PSScriptRoot 'package-submission.ps1')
if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) { throw 'Submission archive was not created.' }
$archiveInfo = Get-Item -LiteralPath $archive
if ($archiveInfo.LastWriteTimeUtc -lt $promotionStarted) { throw 'The submission archive was not refreshed.' }
Write-Output "Validated and packaged the v2 results. Previous files are saved in: $backup"
