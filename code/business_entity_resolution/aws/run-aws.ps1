[CmdletBinding()]
param(
    [string]$Region = "eu-north-1",
    [string]$TrainPrefix,
    [string]$TestPrefix,
    [string]$OutputPrefix,
    [string]$InstanceType = "ml.r6i.4xlarge",
    [int]$VolumeSizeGB = 250,
    [int]$TimeoutMinutes = 720,
    [switch]$SmokeTest,
    [switch]$StageOnly,
    [switch]$NoInputSync,
    [switch]$NoDownload
)

$ErrorActionPreference = "Stop"
$env:AWS_SDK_UA_APP_ID = "AWSSkill-SageMaker"
$env:AWS_DEFAULT_REGION = $Region

function Invoke-AwsCli {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $result = & aws @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "AWS CLI failed (exit $LASTEXITCODE): aws $($Arguments -join ' ')"
    }
    return $result
}

function Write-JsonFile {
    param([string]$Path, [object]$Value)
    $json = $Value | ConvertTo-Json -Depth 30
    [System.IO.File]::WriteAllText($Path, $json, [System.Text.UTF8Encoding]::new($false))
}

function Wait-ProcessingJob {
    param([string]$JobName)
    $terminal = @("Completed", "Failed", "Stopped")
    do {
        Start-Sleep -Seconds 30
        $state = (Invoke-AwsCli @(
            "sagemaker", "describe-processing-job", "--processing-job-name", $JobName,
            "--query", "ProcessingJobStatus", "--output", "text"
        )).Trim()
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] SageMaker Processing $JobName status: $state"
    } while ($state -notin $terminal)

    if ($state -ne "Completed") {
        $details = (Invoke-AwsCli @(
            "sagemaker", "describe-processing-job", "--processing-job-name", $JobName,
            "--output", "json"
        ) | Out-String) | ConvertFrom-Json
        if ($details.FailureReason) { Write-Host "Failure reason: $($details.FailureReason)" }
        throw "SageMaker Processing job $JobName ended with state $state."
    }
}

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$packageRoot = Join-Path $workspace "code\business_entity_resolution"
$scratchRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("devcore-er-sagemaker-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $scratchRoot | Out-Null

$accountId = (Invoke-AwsCli @("sts", "get-caller-identity", "--query", "Account", "--output", "text")).Trim()
if ($Region -ne "eu-north-1") {
    throw "This SageMaker project, Spark image, and quota request are configured for eu-north-1."
}
$projectId = "55r18om2bv9mkz"
$projectBucket = "amazon-sagemaker-$accountId-$Region-$projectId"
$regionalRoot = "s3://$projectBucket/shared/amazonml"
$sourceBucket = "amazonml-er-$accountId-us-east-1-20260926"
$sourceData = "s3://$sourceBucket/input"
$sourceSmoke = "s3://$sourceBucket/smoke"
$codePrefix = "$regionalRoot/code/src"
$roleArn = "arn:aws:iam::${accountId}:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
$imageUri = "330188676905.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-spark-processing:3.5-cpu-py312-v1.4"

if (-not $TrainPrefix) {
    if ($SmokeTest) { $TrainPrefix = "$regionalRoot/smoke/train" }
    else { $TrainPrefix = "$regionalRoot/input/train" }
}
if (-not $TestPrefix) {
    if ($SmokeTest) { $TestPrefix = "$regionalRoot/smoke/test" }
    else { $TestPrefix = "$regionalRoot/input/test" }
}
$runLabel = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $OutputPrefix) {
    $runKind = if ($SmokeTest) { "smoke" } else { "full" }
    $OutputPrefix = "$regionalRoot/output/sagemaker/$runKind-$runLabel"
}
$OutputPrefix = $OutputPrefix.TrimEnd("/")

Invoke-AwsCli @("s3api", "head-bucket", "--bucket", $projectBucket, "--region", $Region) | Out-Null
if (-not $NoInputSync) {
    if ($SmokeTest) {
        Invoke-AwsCli @(
            "s3", "sync", "$sourceSmoke/", "$regionalRoot/smoke/",
            "--source-region", "us-east-1", "--region", $Region, "--only-show-errors"
        ) | Out-Null
    } else {
        Invoke-AwsCli @(
            "s3", "sync", "$sourceData/", "$regionalRoot/input/",
            "--source-region", "us-east-1", "--region", $Region, "--only-show-errors"
        ) | Out-Null
    }
}

Invoke-AwsCli @(
    "s3", "cp", (Join-Path $packageRoot "src\sagemaker_spark_job.py"),
    "$codePrefix/sagemaker_spark_job.py", "--region", $Region, "--only-show-errors"
) | Out-Null
Invoke-AwsCli @(
    "s3", "cp", (Join-Path $packageRoot "src\er_core.py"),
    "$codePrefix/er_core.py", "--region", $Region, "--only-show-errors"
) | Out-Null

$jobName = "devcore-er-$runLabel"
if ($SmokeTest) { $jobName = "devcore-er-smoke-$runLabel" }
$scriptPath = "/opt/ml/processing/input/code/sagemaker_spark_job.py"
$sparkCommand = @(
    "smspark-submit",
    "--local-spark-event-logs-dir", "/opt/ml/processing/spark-events/",
    "--conf", "spark.driver.memory=12g",
    "--conf", "spark.driver.memoryOverhead=2g",
    "--conf", "spark.executor.instances=1",
    "--conf", "spark.executor.cores=12",
    "--conf", "spark.executor.memory=80g",
    "--conf", "spark.executor.memoryOverhead=12g",
    "--conf", "spark.default.parallelism=96",
    "--py-files", "/opt/ml/processing/input/code/er_core.py",
    $scriptPath
)
$job = @{
    ProcessingJobName = $jobName
    RoleArn = $roleArn
    AppSpecification = @{
        ImageUri = $imageUri
        ContainerEntrypoint = $sparkCommand
        ContainerArguments = @(
            "--train-prefix", $TrainPrefix.TrimEnd("/"),
            "--test-prefix", $TestPrefix.TrimEnd("/"),
            "--output-prefix", $OutputPrefix,
            "--smoke-test", $(if ($SmokeTest) { "true" } else { "false" }),
            "--instance-type", $InstanceType
        )
    }
    ProcessingResources = @{
        ClusterConfig = @{
            InstanceCount = 1
            InstanceType = $InstanceType
            VolumeSizeInGB = $VolumeSizeGB
        }
    }
    ProcessingInputs = @(@{
        InputName = "code"
        S3Input = @{
            S3Uri = "$codePrefix/"
            LocalPath = "/opt/ml/processing/input/code"
            S3DataType = "S3Prefix"
            S3InputMode = "File"
            S3DataDistributionType = "FullyReplicated"
            S3CompressionType = "None"
        }
    })
    ProcessingOutputConfig = @{
        Outputs = @(@{
            OutputName = "spark-events"
            S3Output = @{
                S3Uri = "$OutputPrefix/spark-events/"
                LocalPath = "/opt/ml/processing/spark-events"
                S3UploadMode = "EndOfJob"
            }
        })
    }
    StoppingCondition = @{ MaxRuntimeInSeconds = $TimeoutMinutes * 60 }
    Environment = @{ AWS_DEFAULT_REGION = $Region }
    Tags = @(
        @{ Key = "team"; Value = "DevCore" },
        @{ Key = "challenge"; Value = "AmazonML-EntityResolution" }
    )
}
$jobPath = Join-Path $scratchRoot "processing-job.json"
Write-JsonFile $jobPath $job

if ($StageOnly) {
    $null = Invoke-AwsCli @(
        "sagemaker", "create-processing-job", "--cli-input-json", "file://$jobPath",
        "--generate-cli-skeleton", "output", "--region", $Region
    )
    Write-Host "Inputs and SageMaker Spark source are staged in $projectBucket; the Processing request validated and no compute was started."
    exit 0
}

$quotaValue = (Invoke-AwsCli @(
    "service-quotas", "get-service-quota", "--service-code", "sagemaker",
    "--quota-code", "L-49765109", "--region", $Region,
    "--query", "Quota.Value", "--output", "text"
)).Trim()
if ([double]$quotaValue -lt 1) {
    $quotaStatus = (Invoke-AwsCli @(
        "service-quotas", "list-requested-service-quota-change-history",
        "--service-code", "sagemaker", "--region", $Region,
        "--query", "RequestedQuotas[?QuotaCode=='L-49765109'] | [0].Status", "--output", "text"
    )).Trim()
    if (-not $quotaStatus -or $quotaStatus -eq "None") { $quotaStatus = "no request found" }
    throw "SageMaker Processing quota for ml.r6i.4xlarge is $quotaValue in $Region (request status: $quotaStatus). No job was started."
}

Write-Host "Region: $Region; instance: $InstanceType; one node; max runtime: $TimeoutMinutes minutes"
Write-Host "Input train: $TrainPrefix; test: $TestPrefix"
Write-Host "Output prefix: $OutputPrefix"

Invoke-AwsCli @("sagemaker", "create-processing-job", "--cli-input-json", "file://$jobPath", "--region", $Region) | Out-Null
Write-Host "Started SageMaker Processing job: $jobName"
Wait-ProcessingJob $jobName
Write-Host "SageMaker Processing completed: $OutputPrefix"

if ($NoDownload) { exit 0 }

$downloadRoot = Join-Path $scratchRoot "download"
$candidateParts = Join-Path $downloadRoot "candidate_parts"
$matchingParts = Join-Path $downloadRoot "matching_parts"
$metricsParts = Join-Path $downloadRoot "metrics_parts"
New-Item -ItemType Directory -Force -Path $candidateParts, $matchingParts, $metricsParts | Out-Null
Invoke-AwsCli @("s3", "cp", "$OutputPrefix/submission/candidate_pairs/", $candidateParts, "--recursive", "--region", $Region, "--only-show-errors") | Out-Null
Invoke-AwsCli @("s3", "cp", "$OutputPrefix/submission/matching_results/", $matchingParts, "--recursive", "--region", $Region, "--only-show-errors") | Out-Null
Invoke-AwsCli @("s3", "cp", "$OutputPrefix/metrics/", $metricsParts, "--recursive", "--region", $Region, "--only-show-errors") | Out-Null
python (Join-Path $packageRoot "src\materialize_output.py") `
    --candidate-parts $candidateParts `
    --matching-parts $matchingParts `
    --metrics-parts $metricsParts `
    --output-dir (Join-Path $workspace "output")
if ($LASTEXITCODE -ne 0) { throw "Could not materialize SageMaker output files." }
Write-Host "Submission and metrics files are in $(Join-Path $workspace 'output')"
Write-Host "SageMaker job: $jobName; output prefix: $OutputPrefix"
