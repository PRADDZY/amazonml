[CmdletBinding()]
param(
    [string]$Region = "us-east-1",
    [ValidateSet("G.1X", "G.2X", "G.4X")]
    [string]$WorkerType = "G.2X",
    [ValidateRange(2, 100)]
    [int]$NumberOfWorkers = 10,
    [ValidateRange(1, 10080)]
    [int]$TimeoutMinutes = 720,
    [switch]$SmokeTest,
    [switch]$StageOnly,
    [switch]$NoDownload
)

$ErrorActionPreference = "Stop"
$env:AWS_SDK_UA_APP_ID = "AWSSkill-Glue"
$env:AWS_DEFAULT_REGION = $Region

function Invoke-AwsCli {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $result = & aws @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "AWS CLI failed (exit $exitCode): aws $($Arguments -join ' ')`n$(($result | Out-String).Trim())"
    }
    return $result
}

function Wait-GlueJobRun {
    param([string]$JobName, [string]$RunId)
    $terminal = @("SUCCEEDED", "FAILED", "ERROR", "TIMEOUT", "STOPPED", "EXPIRED")
    do {
        Start-Sleep -Seconds 30
        $state = (Invoke-AwsCli @(
            "glue", "get-job-run", "--job-name", $JobName, "--run-id", $RunId,
            "--query", "JobRun.JobRunState", "--output", "text", "--region", $Region
        ) | Out-String).Trim()
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Glue $JobName/$RunId status: $state"
    } while ($state -notin $terminal)

    if ($state -ne "SUCCEEDED") {
        $details = (Invoke-AwsCli @(
            "glue", "get-job-run", "--job-name", $JobName, "--run-id", $RunId,
            "--output", "json", "--region", $Region
        ) | Out-String) | ConvertFrom-Json
        if ($details.JobRun.ErrorMessage) { Write-Host "Failure reason: $($details.JobRun.ErrorMessage)" }
        if ($details.JobRun.LogGroupName) { Write-Host "CloudWatch logs: $($details.JobRun.LogGroupName)" }
        throw "Glue job run $JobName/$RunId ended with state $state."
    }
}

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$packageRoot = Join-Path $workspace "code\business_entity_resolution"
$scratchRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("devcore-er-glue-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $scratchRoot | Out-Null

$accountId = (Invoke-AwsCli @("sts", "get-caller-identity", "--query", "Account", "--output", "text", "--region", $Region) | Out-String).Trim()
if ($Region -ne "us-east-1") {
    throw "The preconfigured Glue role, challenge bucket, and 1,000-DPU quota are in us-east-1."
}

$bucket = "amazonml-er-${accountId}-us-east-1-20260926"
$bucketUri = "s3://$bucket"
$codePrefix = "$bucketUri/code"
$tempPrefix = "$bucketUri/glue-temp/devcore-er"
$roleArn = "arn:aws:iam::${accountId}:role/AmazonMLEntityResolutionGlueRole"
$workerDpu = switch ($WorkerType) { "G.1X" { 1 } "G.2X" { 2 } "G.4X" { 4 } }
$runWorkers = if ($SmokeTest) { [Math]::Min(2, $NumberOfWorkers) } else { $NumberOfWorkers }
$requiredDpu = $workerDpu * $runWorkers
$dpuQuota = [double]((Invoke-AwsCli @(
    "service-quotas", "get-service-quota", "--service-code", "glue",
    "--quota-code", "L-08F3B322", "--region", $Region,
    "--query", "Quota.Value", "--output", "text"
) | Out-String).Trim())
if ($requiredDpu -gt $dpuQuota) {
    throw "This run needs $requiredDpu Glue DPUs but the regional task-DPU quota is $dpuQuota."
}

$runLabel = Get-Date -Format "yyyyMMdd-HHmmss"
$runKind = if ($SmokeTest) { "smoke" } else { "full" }
$jobName = "devcore-er-$runKind-$runLabel"
$runPrefix = "$bucketUri/output/glue/$runKind-$runLabel"
$glueTemp = "$tempPrefix/$runLabel"
$dataKind = if ($SmokeTest) { "smoke" } else { "input" }
$trainPrefix = "$bucketUri/$dataKind/train"
$testPrefix = "$bucketUri/$dataKind/test"
$scriptUri = "$codePrefix/er_job.py"
$coreUri = "$codePrefix/er_core.py"

Invoke-AwsCli @(
    "s3", "cp", (Join-Path $packageRoot "src\sagemaker_spark_job.py"),
    $scriptUri, "--region", $Region, "--only-show-errors"
) | Out-Null
Invoke-AwsCli @(
    "s3", "cp", (Join-Path $packageRoot "src\er_core.py"),
    $coreUri, "--region", $Region, "--only-show-errors"
) | Out-Null

$defaultArguments = @{
    "--TempDir" = "$glueTemp/"
    "--extra-py-files" = $coreUri
    "--enable-auto-scaling" = "true"
    "--enable-continuous-cloudwatch-log" = "true"
    "--enable-job-insights" = "true"
    "--enable-metrics" = "true"
    "--enable-spark-ui" = "true"
    "--job-language" = "python"
    "--spark-event-logs-path" = "$glueTemp/spark-ui/"
}
$job = @{
    Name = $jobName
    Description = "DevCore Amazon ML entity resolution challenge pipeline"
    Role = $roleArn
    Command = @{
        Name = "glueetl"
        ScriptLocation = $scriptUri
        PythonVersion = "3"
    }
    GlueVersion = "5.0"
    WorkerType = $WorkerType
    NumberOfWorkers = $runWorkers
    Timeout = $TimeoutMinutes
    MaxRetries = 0
    ExecutionProperty = @{ MaxConcurrentRuns = 1 }
    DefaultArguments = $defaultArguments
    Tags = @{
        team = "DevCore"
        challenge = "AmazonML-EntityResolution"
    }
}
$jobPath = Join-Path $scratchRoot "glue-job.json"
$jobJson = $job | ConvertTo-Json -Depth 20
[System.IO.File]::WriteAllText($jobPath, $jobJson, [System.Text.UTF8Encoding]::new($false))

if ($StageOnly) {
    $null = Invoke-AwsCli @(
        "glue", "create-job", "--cli-input-json", "file://$jobPath",
        "--generate-cli-skeleton", "output", "--region", $Region
    )
    Write-Host "Glue source staged and job request validated. No compute started."
    Write-Host "Region: $Region; workers: $runWorkers x $WorkerType; DPUs: $requiredDpu/$dpuQuota"
    exit 0
}

$null = Invoke-AwsCli @(
    "glue", "create-job", "--cli-input-json", "file://$jobPath", "--region", $Region
)

$jobArguments = @{
    "--train-prefix" = $trainPrefix
    "--test-prefix" = $testPrefix
    "--output-prefix" = $runPrefix
    "--smoke-test" = $(if ($SmokeTest) { "true" } else { "false" })
    "--instance-type" = "Glue $WorkerType x $runWorkers workers"
    "--compute-service" = "AWS Glue Spark 5.0"
    "--s3-scheme" = "s3"
}
$runRequest = @{
    JobName = $jobName
    Timeout = $TimeoutMinutes
    WorkerType = $WorkerType
    NumberOfWorkers = $runWorkers
    Arguments = $jobArguments
}
$runPath = Join-Path $scratchRoot "glue-run.json"
$runJson = $runRequest | ConvertTo-Json -Depth 20
[System.IO.File]::WriteAllText($runPath, $runJson, [System.Text.UTF8Encoding]::new($false))
$runDetails = (Invoke-AwsCli @(
    "glue", "start-job-run", "--cli-input-json", "file://$runPath", "--region", $Region
) | Out-String) | ConvertFrom-Json
$runId = $runDetails.JobRunId

Write-Host "Started Glue Spark run: $jobName/$runId"
Write-Host "Region: $Region; workers: $runWorkers x $WorkerType; DPUs: $requiredDpu/$dpuQuota"
Write-Host "Input train: $trainPrefix; test: $testPrefix"
Write-Host "Output prefix: $runPrefix"
Wait-GlueJobRun $jobName $runId
Write-Host "Glue Spark completed: $runPrefix"

if ($NoDownload) { exit 0 }

$downloadRoot = Join-Path $scratchRoot "download"
$candidateParts = Join-Path $downloadRoot "candidate_parts"
$matchingParts = Join-Path $downloadRoot "matching_parts"
$metricsParts = Join-Path $downloadRoot "metrics_parts"
New-Item -ItemType Directory -Force -Path $candidateParts, $matchingParts, $metricsParts | Out-Null
Invoke-AwsCli @("s3", "cp", "$runPrefix/submission/candidate_pairs/", $candidateParts, "--recursive", "--region", $Region, "--only-show-errors") | Out-Null
Invoke-AwsCli @("s3", "cp", "$runPrefix/submission/matching_results/", $matchingParts, "--recursive", "--region", $Region, "--only-show-errors") | Out-Null
Invoke-AwsCli @("s3", "cp", "$runPrefix/metrics/", $metricsParts, "--recursive", "--region", $Region, "--only-show-errors") | Out-Null
python (Join-Path $packageRoot "src\materialize_output.py") `
    --candidate-parts $candidateParts `
    --matching-parts $matchingParts `
    --metrics-parts $metricsParts `
    --output-dir (Join-Path $workspace "output")
if ($LASTEXITCODE -ne 0) { throw "Could not materialize the AWS Glue output files." }
Write-Host "Submission and metrics files are in $(Join-Path $workspace 'output')"
Write-Host "Glue job: $jobName/$runId; output prefix: $runPrefix"
