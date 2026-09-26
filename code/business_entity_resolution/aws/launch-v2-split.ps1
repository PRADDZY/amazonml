[CmdletBinding()]
param(
    [ValidateSet('features','fit','test','finalize')][string]$Phase = 'features',
    [string]$RunId,
    [ValidateRange(2, 6)][int]$ShardCount = 6,
    [ValidateRange(1, 20)][int]$MaxHours = 20,
    [switch]$StageOnly
)

$ErrorActionPreference = 'Stop'
$env:AWS_SDK_UA_APP_ID = 'AWSSkill-EC2'
$regionPlan = [ordered]@{
    'us-east-1' = 2
    'eu-north-1' = 2
    'ap-south-1' = 1
    'ca-central-1' = 1
}
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$packageRoot = Join-Path $workspace 'code\business_entity_resolution'
$stage = Join-Path $workspace "output\v2\split-stage-$Phase"
New-Item -ItemType Directory -Force -Path $stage | Out-Null

function Invoke-Aws {
    param([string[]]$CliArguments)
    $result = & aws @CliArguments
    if ($LASTEXITCODE -ne 0) {
        throw "AWS CLI failed (exit $LASTEXITCODE): aws $($CliArguments -join ' ')"
    }
    return $result
}

function Write-Json {
    param([string]$Path, $Value)
    [IO.File]::WriteAllText($Path, (ConvertTo-Json -InputObject $Value -Depth 30), [Text.UTF8Encoding]::new($false))
}

$latestRunPath = Join-Path $workspace 'output\v2\latest-split-run.json'
if (-not $RunId) {
    if ($Phase -eq 'features') {
        $RunId = Get-Date -Format 'yyyyMMdd-HHmmss'
    }
    elseif (Test-Path -LiteralPath $latestRunPath) {
        $RunId = (Get-Content -LiteralPath $latestRunPath -Raw | ConvertFrom-Json).run_id
    }
    else {
        throw 'No split run is recorded. Start with -Phase features or pass -RunId.'
    }
}
$account = (Invoke-Aws @('sts','get-caller-identity','--query','Account','--output','text','--region','us-east-1') | Out-String).Trim()
$bucket = "amazonml-er-$account-us-east-1-20260926"
$bucketUri = "s3://$bucket"
$runUri = "$bucketUri/output/v2/split-$RunId"
$codeUri = "$bucketUri/code/v2/split-$RunId/code.zip"
$localRun = Join-Path $workspace "output\v2\split-$RunId"
New-Item -ItemType Directory -Force -Path $localRun | Out-Null
$existingRunPath = Join-Path $localRun 'run.json'
if (Test-Path -LiteralPath $existingRunPath) {
    $runStartedUtc = (Get-Content -LiteralPath $existingRunPath -Raw | ConvertFrom-Json).run_started_utc
}
else {
    $runStartedUtc = (Get-Date).ToUniversalTime().ToString('o')
}
$archive = Join-Path $localRun 'code.zip'

if ($Phase -eq 'features') {
    Compress-Archive -Path (Join-Path $packageRoot 'src'),(Join-Path $packageRoot 'requirements-v2.txt') `
        -DestinationPath $archive -Force
    $null = Invoke-Aws @('s3','cp',$archive,$codeUri,'--region','us-east-1','--only-show-errors')
}
else {
    $null = Invoke-Aws @('s3api','head-object','--bucket',$bucket,'--key',"code/v2/split-$RunId/code.zip",'--region','us-east-1')
}

if ($Phase -in @('features','test')) {
    $instances = @()
    foreach ($entry in $regionPlan.GetEnumerator()) {
        $region = $entry.Key
        $quota = [double]((Invoke-Aws @('service-quotas','get-service-quota','--service-code','ec2',
            '--quota-code','L-1216C47A','--region',$region,'--query','Quota.Value','--output','text') | Out-String).Trim())
        $runningInstances = (Invoke-Aws @('ec2','describe-instances','--region',$region,
            '--filters','Name=instance-state-name,Values=running,pending',
            '--query','Reservations[].Instances[].InstanceType','--output','text') | Out-String).Trim()
        $usedVcpu = 0
        if ($runningInstances -and $runningInstances -ne 'None') {
            foreach ($instanceTypeName in ($runningInstances -split '\s+')) {
                $vcpuText = (Invoke-Aws @('ec2','describe-instance-types','--region',$region,
                    '--instance-types',$instanceTypeName,'--query','InstanceTypes[0].VCpuInfo.DefaultVCpus','--output','text') | Out-String).Trim()
                $usedVcpu += [double]$vcpuText
            }
        }
        $workerSlots = [Math]::Min([int]$entry.Value, [Math]::Floor(($quota - $usedVcpu) / 4))
        for ($slot = 0; $slot -lt $workerSlots -and $instances.Count -lt $ShardCount; $slot++) {
            $instances += [pscustomobject]@{region=$region; shard=$instances.Count; type='r6i.xlarge'; vcpu=4}
        }
    }
    if ($instances.Count -ne $ShardCount) {
        throw "Found capacity for $($instances.Count)/$ShardCount shard workers across the configured regions."
    }
}
else {
    $singleRegion = 'us-east-1'
    $instanceType = if ($Phase -eq 'fit') { 'r6i.2xlarge' } else { 'r6i.xlarge' }
    $instanceVcpu = if ($Phase -eq 'fit') { 8 } else { 4 }
    $quota = [double]((Invoke-Aws @('service-quotas','get-service-quota','--service-code','ec2',
        '--quota-code','L-1216C47A','--region',$singleRegion,'--query','Quota.Value','--output','text') | Out-String).Trim())
    $runningInstances = (Invoke-Aws @('ec2','describe-instances','--region',$singleRegion,
        '--filters','Name=instance-state-name,Values=running,pending',
        '--query','Reservations[].Instances[].InstanceType','--output','text') | Out-String).Trim()
    $usedVcpu = 0
    if ($runningInstances -and $runningInstances -ne 'None') {
        foreach ($instanceTypeName in ($runningInstances -split '\s+')) {
            $vcpuText = (Invoke-Aws @('ec2','describe-instance-types','--region',$singleRegion,
                '--instance-types',$instanceTypeName,'--query','InstanceTypes[0].VCpuInfo.DefaultVCpus','--output','text') | Out-String).Trim()
            $usedVcpu += [double]$vcpuText
        }
    }
    if (($quota - $usedVcpu) -lt $instanceVcpu) {
        throw "Insufficient available vCPU quota in $singleRegion for $instanceType."
    }
    $instances = @([pscustomobject]@{region=$singleRegion; shard=0; type=$instanceType; vcpu=$instanceVcpu})
}

$lifetimeMinutes = $MaxHours * 60 + 30
$jobSeconds = $MaxHours * 3600
$bootstrapTemplate = @'
#!/bin/bash
set -Eeuo pipefail
export AWS_SDK_UA_APP_ID=AWSSkill-EC2
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec > >(tee -a /var/log/devcore-v2-split.log) 2>&1
RUN_URI='__RUN_URI__'
BUCKET_URI='__BUCKET_URI__'
CODE_URI='__CODE_URI__'
RUN_STARTED_UTC='__RUN_STARTED_UTC__'
export DEVCORE_RUN_STARTED_UTC="$RUN_STARTED_UTC"
PHASE='__PHASE__'
SHARD_INDEX='__SHARD_INDEX__'
SHARD_COUNT='__SHARD_COUNT__'
WORKERS='__WORKERS__'
OUTPUT_URI='__OUTPUT_URI__'
shutdown -h +__LIFETIME_MINUTES__
SYNC_PID=''
status() { printf '%s\n' "$1" >/tmp/devcore-v2-split-status.txt; aws s3 cp /tmp/devcore-v2-split-status.txt "$RUN_URI/$PHASE/shard-$(printf '%03d' "$SHARD_INDEX")/status.txt" --region us-east-1 --only-show-errors; }
finish() {
  rc=$?
  set +e
  [ -z "$SYNC_PID" ] || kill "$SYNC_PID" 2>/dev/null
  timeout 900 aws s3 sync /opt/devcore-v2/results/ "$OUTPUT_URI/" --region us-east-1 --only-show-errors
  timeout 120 aws s3 cp /var/log/devcore-v2-split.log "$RUN_URI/$PHASE/shard-$(printf '%03d' "$SHARD_INDEX")/job.log" --region us-east-1 --only-show-errors
  if [ "$rc" -eq 0 ]; then status SUCCEEDED; else status "FAILED exit=$rc"; fi
  shutdown -h now
}
trap finish EXIT
trap 'rc=$?; echo "BOOTSTRAP ERROR exit=$rc line=$LINENO command=$BASH_COMMAND" >&2; exit "$rc"' ERR
status BOOTSTRAPPING
(while sleep 60; do aws s3 cp /var/log/devcore-v2-split.log "$RUN_URI/$PHASE/shard-$(printf '%03d' "$SHARD_INDEX")/job.log" --region us-east-1 --only-show-errors >/dev/null 2>&1 || true; done) &
SYNC_PID=$!
echo STEP_INSTALL_OS_PACKAGES
dnf install -y python3.11 python3.11-pip libgomp
echo STEP_DOWNLOAD_CODE
mkdir -p /opt/devcore-v2/{code,site-packages,data/train,data/test,index,results,fit,feature-shards,test-shards}
aws s3 cp "$CODE_URI" /opt/devcore-v2/code.zip --region us-east-1 --only-show-errors
python3.11 - <<'PY'
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from zipfile import ZipFile
root = Path('/opt/devcore-v2/code')
with ZipFile('/opt/devcore-v2/code.zip') as archive:
    for member in archive.infolist():
        relative = PurePosixPath(member.filename.replace('\\', '/'))
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe ZIP member path: {member.filename!r}')
        target = root.joinpath(*relative.parts)
        if member.is_dir(): target.mkdir(parents=True, exist_ok=True); continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, target.open('wb') as destination: copyfileobj(source, destination)
PY
echo STEP_INSTALL_PYTHON_PACKAGES
python3.11 -m pip install --no-cache-dir --target=/opt/devcore-v2/site-packages -r /opt/devcore-v2/code/requirements-v2.txt
export PYTHONPATH=/opt/devcore-v2/site-packages
python3.11 /opt/devcore-v2/code/src/v2_cloud_smoke.py
status DOWNLOADING_DATA
if [ "$PHASE" = features ] || [ "$PHASE" = fit ]; then
  aws s3 sync "$BUCKET_URI/input/train/" /opt/devcore-v2/data/train/ --region us-east-1 --only-show-errors
  aws s3 sync "$BUCKET_URI/output/v2/cache/unicode-bm25-v2/train/" /opt/devcore-v2/index/ --region us-east-1 --only-show-errors
fi
if [ "$PHASE" = fit ] || [ "$PHASE" = test ] || [ "$PHASE" = finalize ]; then
  aws s3 sync "$BUCKET_URI/input/test/" /opt/devcore-v2/data/test/ --region us-east-1 --only-show-errors
fi
if [ "$PHASE" = features ]; then
  status RUNNING_FEATURE_SHARD
  timeout __JOB_SECONDS__ python3.11 /opt/devcore-v2/code/src/v2_pipeline.py \
    --phase features --train-dir /opt/devcore-v2/data/train --test-dir /opt/devcore-v2/data/test \
    --index-dir /opt/devcore-v2/index --output-dir /opt/devcore-v2/results \
    --workers "$WORKERS" --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT"
elif [ "$PHASE" = fit ]; then
  status MERGING_FEATURES_AND_FITTING
  aws s3 sync "$RUN_URI/feature-shards/" /opt/devcore-v2/feature-shards/ --region us-east-1 --only-show-errors
  timeout __JOB_SECONDS python3.11 /opt/devcore-v2/code/src/v2_pipeline.py \
    --phase fit --train-dir /opt/devcore-v2/data/train --test-dir /opt/devcore-v2/data/test \
    --index-dir /opt/devcore-v2/index --output-dir /opt/devcore-v2/results \
    --feature-dir /opt/devcore-v2/feature-shards --workers "$WORKERS" --shard-count "$SHARD_COUNT"
elif [ "$PHASE" = test ]; then
  status RUNNING_TEST_SHARD
  aws s3 sync "$RUN_URI/fit/" /opt/devcore-v2/fit/ --region us-east-1 --only-show-errors
  timeout __JOB_SECONDS python3.11 /opt/devcore-v2/code/src/v2_pipeline.py \
    --phase test --train-dir /opt/devcore-v2/data/train --test-dir /opt/devcore-v2/data/test \
    --index-dir /opt/devcore-v2/fit/test-index --output-dir /opt/devcore-v2/results \
    --fit-dir /opt/devcore-v2/fit --workers "$WORKERS" --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT"
else
  status MERGING_TEST_SHARDS
  aws s3 sync "$RUN_URI/fit/" /opt/devcore-v2/fit/ --region us-east-1 --only-show-errors
  aws s3 sync "$RUN_URI/test-shards/" /opt/devcore-v2/results/test-shards/ --region us-east-1 --only-show-errors
  timeout __JOB_SECONDS python3.11 /opt/devcore-v2/code/src/v2_pipeline.py \
    --phase finalize --train-dir /opt/devcore-v2/data/train --test-dir /opt/devcore-v2/data/test \
    --index-dir /opt/devcore-v2/fit/test-index --output-dir /opt/devcore-v2/results \
    --fit-dir /opt/devcore-v2/fit --workers "$WORKERS" --shard-count "$SHARD_COUNT"
  rm -rf /opt/devcore-v2/results/test-shards
fi
status UPLOADING_RESULTS
'@

$regionArtifacts = @{}
foreach ($region in ($instances.region | Select-Object -Unique)) {
    $vpc = (Invoke-Aws @('ec2','describe-vpcs','--region',$region,'--filters','Name=is-default,Values=true',
        '--query','Vpcs[0].VpcId','--output','text') | Out-String).Trim()
    if (-not $vpc -or $vpc -eq 'None') { throw "No default VPC is available in $region." }
    $offerings = (Invoke-Aws @('ec2','describe-instance-type-offerings','--region',$region,'--location-type','availability-zone',
        '--filters','Name=instance-type,Values=r6i.xlarge,r6i.2xlarge','--query','InstanceTypeOfferings[].Location','--output','text') | Out-String).Trim() -split '\s+'
    $subnetRows = (Invoke-Aws @('ec2','describe-subnets','--region',$region,'--filters',"Name=vpc-id,Values=$vpc",'Name=default-for-az,Values=true',
        '--query','Subnets[].{Id:SubnetId,Az:AvailabilityZone}','--output','json') | Out-String) | ConvertFrom-Json
    $subnet = $subnetRows | Where-Object { $_.Az -in $offerings } | Select-Object -First 1
    $group = (Invoke-Aws @('ec2','describe-security-groups','--region',$region,'--filters',"Name=vpc-id,Values=$vpc",'Name=group-name,Values=default',
        '--query','SecurityGroups[0].GroupId','--output','text') | Out-String).Trim()
    $ami = (Invoke-Aws @('ssm','get-parameter','--region',$region,'--name','/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64',
        '--query','Parameter.Value','--output','text') | Out-String).Trim()
    if (-not $subnet -or $group -eq 'None' -or $ami -eq 'None') { throw "Missing EC2 prerequisites in $region." }
    $regionArtifacts[$region] = @{subnet=$subnet.Id; group=$group; ami=$ami}
}

$workerManifest = @()
foreach ($worker in $instances) {
    $shard = [int]$worker.shard
    $shardName = '{0:D3}' -f $shard
    $region = $worker.region
    $outputUri = switch ($Phase) {
        'features' { "$runUri/feature-shards/$shardName" }
        'fit' { "$runUri/fit" }
        'test' { "$runUri/test-shards/$shardName" }
        'finalize' { "$runUri/final" }
    }
    $bootstrap = $bootstrapTemplate
    $bootstrap = $bootstrap.Replace('__RUN_URI__',$runUri).Replace('__BUCKET_URI__',$bucketUri)
    $bootstrap = $bootstrap.Replace('__CODE_URI__',$codeUri).Replace('__PHASE__',$Phase)
    $bootstrap = $bootstrap.Replace('__RUN_STARTED_UTC__',$runStartedUtc)
    $bootstrap = $bootstrap.Replace('__SHARD_INDEX__',[string]$shard).Replace('__SHARD_COUNT__',[string]$ShardCount)
    $bootstrap = $bootstrap.Replace('__WORKERS__',[string]([Math]::Max(1,[int]$worker.vcpu)))
    $bootstrap = $bootstrap.Replace('__OUTPUT_URI__',$outputUri).Replace('__LIFETIME_MINUTES__',[string]$lifetimeMinutes)
    $bootstrap = $bootstrap.Replace('__JOB_SECONDS__',[string]$jobSeconds)
    $bootstrapPath = Join-Path $stage "bootstrap-$Phase-$shardName.sh"
    [IO.File]::WriteAllText($bootstrapPath,($bootstrap -replace "`r`n","`n"),[Text.UTF8Encoding]::new($false))
    $blocksPath = Join-Path $stage "blocks-$shardName.json"
    $volumeSize = if ($Phase -eq 'fit') { 250 } else { 200 }
    Write-Json $blocksPath @(@{DeviceName='/dev/xvda';Ebs=@{VolumeSize=$volumeSize;VolumeType='gp3';Encrypted=$true;DeleteOnTermination=$true}})
    $tagsPath = Join-Path $stage "tags-$shardName.json"
    $name = "devcore-v2-$Phase-$RunId-$shardName"
    Write-Json $tagsPath @(@{ResourceType='instance';Tags=@(@{Key='Name';Value=$name},@{Key='Team';Value='DevCore'},@{Key='RunId';Value=$RunId},@{Key='Phase';Value=$Phase},@{Key='Shard';Value=$shardName})},@{ResourceType='volume';Tags=@(@{Key='Team';Value='DevCore'},@{Key='RunId';Value=$RunId})})
    if ($StageOnly) {
        Write-Output "Staged $Phase shard $shardName in $region ($($worker.type), $($worker.vcpu) vCPU)"
        continue
    }
    $response = (Invoke-Aws @('ec2','run-instances','--region',$region,'--image-id',$regionArtifacts[$region].ami,
        '--instance-type',$worker.type,'--count','1','--subnet-id',$regionArtifacts[$region].subnet,
        '--security-group-ids',$regionArtifacts[$region].group,'--associate-public-ip-address',
        '--iam-instance-profile','Name=DevCoreEntityResolutionEc2Profile','--user-data',"file://$bootstrapPath",
        '--block-device-mappings',"file://$blocksPath",'--tag-specifications',"file://$tagsPath",
        '--metadata-options','HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=1',
        '--instance-initiated-shutdown-behavior','terminate','--output','json') | Out-String) | ConvertFrom-Json
    $record = [pscustomobject]@{phase=$Phase;shard_index=$shard;shard_count=$ShardCount;region=$region;
        instance_id=$response.Instances[0].InstanceId;instance_type=$worker.type;workers=$worker.vcpu;
        status_uri="$runUri/$Phase/shard-$shardName/status.txt";output_uri=$outputUri}
    $workerManifest += $record
    Write-Output ($record | ConvertTo-Json -Compress)
}

if (-not $StageOnly) {
    $manifestPath = Join-Path $localRun "$Phase-workers.json"
    Write-Json $manifestPath $workerManifest
    $runRecord = [pscustomobject]@{run_id=$RunId;run_uri=$runUri;code_uri=$codeUri;shard_count=$ShardCount;
        run_started_utc=$runStartedUtc;
        last_phase=$Phase;phase_started_utc=(Get-Date).ToUniversalTime().ToString('o');workers=$workerManifest}
    Write-Json $latestRunPath $runRecord
    Write-Json (Join-Path $localRun 'run.json') $runRecord
    $null = Invoke-Aws @('s3','cp',$manifestPath,"$runUri/control/$Phase-workers.json",'--region','us-east-1','--only-show-errors')
    Write-Output "Run $RunId; phase $Phase; output $runUri; workers $($workerManifest.Count)."
}
