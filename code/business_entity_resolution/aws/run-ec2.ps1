[CmdletBinding()]
param(
    [string]$Region = "us-east-1",
    [switch]$SmokeTest,
    [switch]$StageOnly
)

$ErrorActionPreference = "Stop"
$env:AWS_SDK_UA_APP_ID = "AWSSkill-EC2"
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

function Write-JsonFile {
    param([string]$Path, [object]$Value)
    $json = ConvertTo-Json -InputObject $Value -Depth 30
    [System.IO.File]::WriteAllText($Path, $json, [System.Text.UTF8Encoding]::new($false))
}

function Get-RemoteStatus {
    param([string]$Bucket, [string]$Key, [string]$LocalPath)
    try {
        $null = Invoke-AwsCli @(
            "s3", "cp", "s3://$Bucket/$Key", $LocalPath,
            "--region", $Region, "--only-show-errors"
        )
        return (Get-Content -LiteralPath $LocalPath -Raw).Trim()
    }
    catch {
        return ""
    }
}

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$packageRoot = Join-Path $workspace "code\business_entity_resolution"
$awsRoot = Join-Path $packageRoot "aws"
$scratchRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("devcore-er-ec2-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $scratchRoot | Out-Null

$accountId = (Invoke-AwsCli @("sts", "get-caller-identity", "--query", "Account", "--output", "text", "--region", $Region) | Out-String).Trim()
if ($Region -ne "us-east-1") {
    throw "The challenge data, IAM policy scope, and available EC2 quota are in us-east-1."
}
$bucket = "amazonml-er-${accountId}-us-east-1-20260926"
$bucketUri = "s3://$bucket"
$codePrefix = "$bucketUri/code"
$runLabel = Get-Date -Format "yyyyMMdd-HHmmss"
$runKind = if ($SmokeTest) { "smoke" } else { "full" }
$runPrefixKey = "output/glue/ec2-$runKind-$runLabel"
$runUri = "$bucketUri/$runPrefixKey"
$statusKey = "$runPrefixKey/status.txt"
$instanceType = if ($SmokeTest) { "r6i.xlarge" } else { "r6i.2xlarge" }
$workerCount = if ($SmokeTest) { 4 } else { 8 }
$driverMemory = if ($SmokeTest) { "22g" } else { "44g" }
$volumeSize = if ($SmokeTest) { 100 } else { 200 }
$dataKind = if ($SmokeTest) { "smoke" } else { "input" }

$quota = [double]((Invoke-AwsCli @(
    "service-quotas", "get-service-quota", "--service-code", "ec2",
    "--quota-code", "L-1216C47A", "--region", $Region,
    "--query", "Quota.Value", "--output", "text"
) | Out-String).Trim())
$activeVcpu = (Invoke-AwsCli @(
    "ec2", "describe-instances", "--region", $Region,
    "--filters", "Name=instance-state-name,Values=running,pending",
    "--query", "length(Reservations[].Instances[]) ", "--output", "text"
) | Out-String).Trim()
if ($quota -lt $workerCount) {
    throw "The $instanceType run needs $workerCount standard EC2 vCPUs; the regional quota is $quota."
}
if ([int]$activeVcpu -gt 0) {
    throw "Found $activeVcpu running or pending EC2 instances in $Region. The launcher will not risk exceeding the 8-vCPU limit."
}

Invoke-AwsCli @(
    "s3", "cp", (Join-Path $packageRoot "src\sagemaker_spark_job.py"),
    "$codePrefix/er_job.py", "--region", $Region, "--only-show-errors"
) | Out-Null
Invoke-AwsCli @(
    "s3", "cp", (Join-Path $packageRoot "src\er_core.py"),
    "$codePrefix/er_core.py", "--region", $Region, "--only-show-errors"
) | Out-Null

$userDataTemplate = @'
#!/bin/bash
set -Eeuo pipefail
exec > >(tee -a /var/log/devcore-er.log) 2>&1

BUCKET_URI='__BUCKET_URI__'
OUTPUT_URI='__OUTPUT_URI__'
STATUS_URI='__STATUS_URI__'
DATA_KIND='__DATA_KIND__'
SMOKE_TEST='__SMOKE_TEST__'
DRIVER_MEMORY='__DRIVER_MEMORY__'
THREADS='__THREADS__'
LOG_SYNC_PID=''

write_status() {
  printf '%s\n' "$1" > /tmp/devcore-er-status.txt
  aws s3 cp /tmp/devcore-er-status.txt "$STATUS_URI" --only-show-errors --region us-east-1
}

finish() {
  rc=$?
  set +e
  if [ "$rc" -eq 0 ]; then state='SUCCEEDED'; else state="FAILED (exit $rc)"; fi
  printf '%s\n' "$state" > /tmp/devcore-er-status.txt
  if [ -n "$LOG_SYNC_PID" ]; then kill "$LOG_SYNC_PID" 2>/dev/null; fi
  aws s3 cp /var/log/devcore-er.log "$OUTPUT_URI/logs/job.log" --only-show-errors --region us-east-1
  aws s3 cp /tmp/devcore-er-status.txt "$STATUS_URI" --only-show-errors --region us-east-1
  shutdown -h now
}
trap finish EXIT

write_status 'BOOTSTRAPPING'
(while sleep 45; do aws s3 cp /var/log/devcore-er.log "$OUTPUT_URI/logs/job.log" --only-show-errors --region us-east-1 >/dev/null 2>&1 || true; done) &
LOG_SYNC_PID=$!

dnf install -y java-17-amazon-corretto-headless python3.11 python3.11-pip awscli-2
python3.11 -m venv /opt/devcore-venv
/opt/devcore-venv/bin/python -m pip install --no-cache-dir pyspark==3.5.4 numpy==1.26.4 pandas==2.2.3 pyarrow==18.1.0
/opt/devcore-venv/bin/python -c "import numpy, pandas, pyarrow; from pyspark.ml.classification import GBTClassifier; from pyspark.ml.functions import vector_to_array; print('Spark ML dependencies ready')"
export PYSPARK_PYTHON=/opt/devcore-venv/bin/python
export PYSPARK_DRIVER_PYTHON=/opt/devcore-venv/bin/python
export SPARK_LOCAL_DIRS=/opt/devcore-spark-tmp
mkdir -p /opt/devcore/app/data/train /opt/devcore/app/data/test /opt/devcore/app/output /opt/devcore-spark-tmp

aws s3 cp "$BUCKET_URI/code/er_job.py" /opt/devcore/app/er_job.py --only-show-errors --region us-east-1
aws s3 cp "$BUCKET_URI/code/er_core.py" /opt/devcore/app/er_core.py --only-show-errors --region us-east-1
aws s3 sync "$BUCKET_URI/$DATA_KIND/train/" /opt/devcore/app/data/train/ --only-show-errors --region us-east-1
aws s3 sync "$BUCKET_URI/$DATA_KIND/test/" /opt/devcore/app/data/test/ --only-show-errors --region us-east-1
write_status 'RUNNING_SPARK'

timeout 43200 /opt/devcore-venv/bin/spark-submit \
  --master "local[$THREADS]" \
  --driver-memory "$DRIVER_MEMORY" \
  --conf spark.driver.maxResultSize=4g \
  --conf spark.sql.shuffle.partitions=200 \
  --conf spark.local.dir=/opt/devcore-spark-tmp \
  --py-files /opt/devcore/app/er_core.py \
  /opt/devcore/app/er_job.py \
  --train-prefix /opt/devcore/app/data/train \
  --test-prefix /opt/devcore/app/data/test \
  --output-prefix /opt/devcore/app/output \
  --smoke-test "$SMOKE_TEST" \
  --instance-type '__INSTANCE_TYPE__' \
  --compute-service 'Amazon EC2 Spark 3.5.4' \
  --s3-scheme s3

write_status 'UPLOADING_RESULTS'
aws s3 cp /opt/devcore/app/output/ "$OUTPUT_URI/" --recursive --only-show-errors --region us-east-1
write_status 'SUCCEEDED'
'@
$userData = $userDataTemplate.
    Replace("__BUCKET_URI__", $bucketUri).
    Replace("__OUTPUT_URI__", $runUri).
    Replace("__STATUS_URI__", "$bucketUri/$statusKey").
    Replace("__DATA_KIND__", $dataKind).
    Replace("__SMOKE_TEST__", $(if ($SmokeTest) { "true" } else { "false" })).
    Replace("__DRIVER_MEMORY__", $driverMemory).
    Replace("__THREADS__", [string]$workerCount).
    Replace("__INSTANCE_TYPE__", $instanceType)

$userDataPath = Join-Path $scratchRoot "user-data.sh"
$userData = [regex]::Replace($userData, "`r`n|`r", "`n")
[System.IO.File]::WriteAllText($userDataPath, $userData, [System.Text.UTF8Encoding]::new($false))

if ($StageOnly) {
    Write-Host "Bootstrap and source staged. No IAM or EC2 resource was created."
    Write-Host "Region: $Region; instance: $instanceType; vCPUs: $workerCount; RAM: $(if ($SmokeTest) { 32 } else { 64 }) GiB"
    Write-Host "S3 output: $runUri"
    exit 0
}

$roleName = "DevCoreEntityResolutionEc2Role"
$profileName = "DevCoreEntityResolutionEc2Profile"
$trustPolicy = Join-Path $awsRoot "ec2-role-trust.json"
$permissionPolicyTemplate = Get-Content -LiteralPath (Join-Path $awsRoot "ec2-role-policy.json") -Raw
$permissionPolicyJson = $permissionPolicyTemplate.Replace("ACCOUNT", $accountId)
$permissionPolicy = Join-Path $scratchRoot "ec2-role-policy.json"
[System.IO.File]::WriteAllText($permissionPolicy, $permissionPolicyJson, [System.Text.UTF8Encoding]::new($false))
try {
    $null = Invoke-AwsCli @("iam", "get-role", "--role-name", $roleName, "--query", "Role.RoleName", "--output", "text")
}
catch {
    if ($_ -notmatch "NoSuchEntity") { throw }
    $null = Invoke-AwsCli @(
        "iam", "create-role", "--role-name", $roleName,
        "--description", "DevCore entity resolution EC2 runtime with S3-prefix-scoped access",
        "--assume-role-policy-document", "file://$trustPolicy",
        "--tags", "Key=Team,Value=DevCore", "Key=Challenge,Value=AmazonML-EntityResolution"
    )
}
Invoke-AwsCli @(
    "iam", "put-role-policy", "--role-name", $roleName,
    "--policy-name", "DevCoreEntityResolutionS3Access",
    "--policy-document", "file://$permissionPolicy"
) | Out-Null

try {
    $null = Invoke-AwsCli @("iam", "get-instance-profile", "--instance-profile-name", $profileName)
}
catch {
    if ($_ -notmatch "NoSuchEntity") { throw }
    $null = Invoke-AwsCli @(
        "iam", "create-instance-profile", "--instance-profile-name", $profileName,
        "--tags", "Key=Team,Value=DevCore", "Key=Challenge,Value=AmazonML-EntityResolution"
    )
}
$profileRoles = (Invoke-AwsCli @(
    "iam", "get-instance-profile", "--instance-profile-name", $profileName,
    "--query", "InstanceProfile.Roles[].RoleName", "--output", "text"
) | Out-String).Trim()
if ($profileRoles -notmatch [regex]::Escape($roleName)) {
    $null = Invoke-AwsCli @(
        "iam", "add-role-to-instance-profile", "--instance-profile-name", $profileName,
        "--role-name", $roleName
    )
    Start-Sleep -Seconds 15
}

$subnetId = (Invoke-AwsCli @(
    "ec2", "describe-subnets", "--region", $Region,
    "--filters", "Name=vpc-id,Values=vpc-04c41e12daa6b5411", "Name=map-public-ip-on-launch,Values=true",
    "--query", "Subnets[?AvailabilityZone=='us-east-1a'] | [0].SubnetId", "--output", "text"
) | Out-String).Trim()
$securityGroupId = (Invoke-AwsCli @(
    "ec2", "describe-security-groups", "--region", $Region,
    "--filters", "Name=vpc-id,Values=vpc-04c41e12daa6b5411", "Name=group-name,Values=default",
    "--query", "SecurityGroups[0].GroupId", "--output", "text"
) | Out-String).Trim()
$amiId = (Invoke-AwsCli @(
    "ssm", "get-parameter", "--name", "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64",
    "--region", $Region, "--query", "Parameter.Value", "--output", "text"
) | Out-String).Trim()
if (-not $subnetId -or $subnetId -eq "None" -or -not $securityGroupId -or -not $amiId) {
    throw "Could not resolve the public subnet, default security group, or Amazon Linux AMI in $Region."
}

$blockPath = Join-Path $scratchRoot "block-device.json"
Write-JsonFile $blockPath @(@{
    DeviceName = "/dev/xvda"
    Ebs = @{ VolumeSize = $volumeSize; VolumeType = "gp3"; DeleteOnTermination = $true; Encrypted = $true }
})
$tagPath = Join-Path $scratchRoot "tags.json"
$nameTag = "devcore-er-$runKind-$runLabel"
$tags = @(
    @{ ResourceType = "instance"; Tags = @(
        @{ Key = "Name"; Value = $nameTag }, @{ Key = "Team"; Value = "DevCore" },
        @{ Key = "Challenge"; Value = "AmazonML-EntityResolution" }
    ) },
    @{ ResourceType = "volume"; Tags = @(
        @{ Key = "Name"; Value = $nameTag }, @{ Key = "Team"; Value = "DevCore" },
        @{ Key = "Challenge"; Value = "AmazonML-EntityResolution" }
    ) }
)
Write-JsonFile $tagPath $tags

$instanceJson = (Invoke-AwsCli @(
    "ec2", "run-instances", "--image-id", $amiId, "--instance-type", $instanceType,
    "--count", "1", "--subnet-id", $subnetId, "--security-group-ids", $securityGroupId,
    "--associate-public-ip-address", "--iam-instance-profile", "Name=$profileName",
    "--user-data", "file://$userDataPath", "--block-device-mappings", "file://$blockPath",
    "--tag-specifications", "file://$tagPath",
    "--metadata-options", "HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=1",
    "--instance-initiated-shutdown-behavior", "terminate",
    "--region", $Region, "--output", "json"
) | Out-String) | ConvertFrom-Json
$instanceId = $instanceJson.Instances[0].InstanceId
Write-Host "Started $instanceId ($instanceType) in $Region. Output: $runUri"

$statusLocalPath = Join-Path $scratchRoot "status.txt"
$lastStatus = ""
while ($true) {
    Start-Sleep -Seconds 30
    $status = Get-RemoteStatus $bucket $statusKey $statusLocalPath
    if ($status -and $status -ne $lastStatus) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] EC2 Spark status: $status"
        $lastStatus = $status
    }
    if ($status -eq "SUCCEEDED") { break }
    if ($status -like "FAILED*") {
        try { Invoke-AwsCli @("ec2", "terminate-instances", "--instance-ids", $instanceId, "--region", $Region) | Out-Null } catch {}
        throw "AWS EC2 Spark run failed. Log: $runUri/logs/job.log"
    }
    $state = (Invoke-AwsCli @(
        "ec2", "describe-instances", "--instance-ids", $instanceId, "--region", $Region,
        "--query", "Reservations[0].Instances[0].State.Name", "--output", "text"
    ) | Out-String).Trim()
    if ($state -in @("terminated", "stopped")) {
        throw "Instance $instanceId entered state $state before reporting success. Check $runUri/logs/job.log and EC2 console output."
    }
    if ($state -eq "running") {
        $logPath = Join-Path $scratchRoot "job.log"
        try {
            $null = Invoke-AwsCli @(
                "s3", "cp", "$runUri/logs/job.log", $logPath,
                "--region", $Region, "--only-show-errors"
            )
            $tail = Get-Content -LiteralPath $logPath -Tail 3
            if ($tail) { $tail | ForEach-Object { Write-Host "  $_" } }
        }
        catch { }
    }
}

Write-Host "AWS EC2 Spark run succeeded: $runUri"
if ($SmokeTest) { exit 0 }

$downloadRoot = Join-Path $scratchRoot "download"
$candidateParts = Join-Path $downloadRoot "candidate_parts"
$matchingParts = Join-Path $downloadRoot "matching_parts"
$metricsParts = Join-Path $downloadRoot "metrics_parts"
New-Item -ItemType Directory -Force -Path $candidateParts, $matchingParts, $metricsParts | Out-Null
Invoke-AwsCli @("s3", "cp", "$runUri/submission/candidate_pairs/", $candidateParts, "--recursive", "--region", $Region, "--only-show-errors") | Out-Null
Invoke-AwsCli @("s3", "cp", "$runUri/submission/matching_results/", $matchingParts, "--recursive", "--region", $Region, "--only-show-errors") | Out-Null
Invoke-AwsCli @("s3", "cp", "$runUri/metrics/", $metricsParts, "--recursive", "--region", $Region, "--only-show-errors") | Out-Null
python (Join-Path $packageRoot "src\materialize_output.py") `
    --candidate-parts $candidateParts `
    --matching-parts $matchingParts `
    --metrics-parts $metricsParts `
    --output-dir (Join-Path $workspace "output")
if ($LASTEXITCODE -ne 0) { throw "Could not materialize the AWS EC2 Spark output files." }
Write-Host "Submission and metrics files are in $(Join-Path $workspace 'output')"
Write-Host "Output prefix: $runUri"
