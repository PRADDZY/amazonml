$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$studentRoot = Join-Path $workspace "6ab10eb3b23ba_student_resource\student_resource"
$packageRoot = Join-Path $workspace "code\business_entity_resolution"
$outputRoot = Join-Path $workspace "output"
$validator = Join-Path $studentRoot "utils\validate_submission.py"
$pythonOutput = Join-Path $outputRoot "matching_results.tsv"
$candidateOutput = Join-Path $outputRoot "candidate_pairs.tsv"

Push-Location $studentRoot
try {
    python $validator --matching $pythonOutput --candidate $candidateOutput --test-dir (Join-Path $studentRoot "dataset\test")
    if ($LASTEXITCODE -ne 0) { throw "Submission validation failed; archive was not created." }
}
finally {
    Pop-Location
}

$stage = Join-Path ([System.IO.Path]::GetTempPath()) ("amazonml-submission-" + [guid]::NewGuid().ToString("N"))
$bundle = Join-Path $stage "bundle"
New-Item -ItemType Directory -Force -Path (Join-Path $bundle "output") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $bundle "code") | Out-Null
Copy-Item -LiteralPath $pythonOutput -Destination (Join-Path $bundle "output\matching_results.tsv")
Copy-Item -LiteralPath $candidateOutput -Destination (Join-Path $bundle "output\candidate_pairs.tsv")
$codeDestination = Join-Path $bundle "code\business_entity_resolution"
New-Item -ItemType Directory -Force -Path $codeDestination | Out-Null
foreach ($sourceFile in Get-ChildItem -LiteralPath $packageRoot -Recurse -File) {
    if ($sourceFile.FullName -match "\\__pycache__\\" -or $sourceFile.Name -like "*.pyc" -or $sourceFile.Name -eq "sagemaker-trust-policy.json" -or $sourceFile.Name -eq "run-glue.ps1") { continue }
    $relativePath = $sourceFile.FullName.Substring($packageRoot.Length).TrimStart("\")
    $targetFile = Join-Path $codeDestination $relativePath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $targetFile) | Out-Null
    Copy-Item -LiteralPath $sourceFile.FullName -Destination $targetFile
}
Copy-Item -LiteralPath (Join-Path $workspace "Documentation_template.md") -Destination $bundle

$archive = Join-Path $workspace "DevCore_submission.zip"
Compress-Archive -Path (Join-Path $bundle "*") -DestinationPath $archive -CompressionLevel Optimal -Force
Write-Host "Validated submission archive: $archive"
