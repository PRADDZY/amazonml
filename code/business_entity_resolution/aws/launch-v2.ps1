[CmdletBinding()]
param(
    [ValidateSet('probe','full')][string]$Mode = 'probe',
    [ValidateRange(1, 12)][int]$MaxHours = 3,
    [ValidateRange(1, 10000)][int]$SampleModulus = 500,
    [switch]$StageOnly
)
$ErrorActionPreference = 'Stop'
$env:AWS_SDK_UA_APP_ID = 'AWSSkill-SageMaker'
$region = 'us-east-1'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$packageRoot = Join-Path $workspace 'code\business_entity_resolution'
$label = Get-Date -Format 'yyyyMMdd-HHmmss'
$stage = Join-Path $workspace "output\v2\$Mode-$label"
New-Item -ItemType Directory -Force -Path $stage | Out-Null

function Invoke-Aws {
    param([string[]]$CliArguments)
    $result = & aws @CliArguments
    if ($LASTEXITCODE -ne 0) { throw "AWS CLI failed: $($CliArguments[0]) $($CliArguments[1])" }
    return $result
}
function Write-Json {
    param([string]$Path, $Value)
    [IO.File]::WriteAllText($Path, (ConvertTo-Json -InputObject $Value -Depth 20), [Text.UTF8Encoding]::new($false))
}

$account = (Invoke-Aws @('sts','get-caller-identity','--query','Account','--output','text','--region',$region) | Out-String).Trim()
$bucket = "amazonml-er-$account-us-east-1-20260926"
$bucketUri = "s3://$bucket"
$runUri = "$bucketUri/output/v2/$Mode-$label"
$codeUri = "$bucketUri/code/v2/$label/code.zip"
$cacheUri = "$bucketUri/output/v2/cache/unicode-bm25-v2/train"
$archive = Join-Path $stage 'code.zip'
Compress-Archive -Path (Join-Path $packageRoot 'src'),(Join-Path $packageRoot 'requirements-v2.txt') -DestinationPath $archive
$null = Invoke-Aws @('s3','cp',$archive,$codeUri,'--region',$region,'--only-show-errors')
$lifetimeMinutes = $MaxHours * 60
$jobSeconds = $MaxHours * 3600 - 900
$bootstrap = @'
#!/bin/bash
set -Eeuo pipefail
export AWS_SDK_UA_APP_ID=AWSSkill-SageMaker
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec > >(tee -a /var/log/devcore-v2.log) 2>&1
RUN_URI='__RUN__'
BUCKET_URI='__BUCKET__'
CACHE_URI='__CACHE__'
SYNC_PID=''
mkdir -p /opt/devcore-v2/{data/train,index,results,code}
# An independent shutdown timer covers installation hangs and killed bootstrap processes.
shutdown -h +__MINUTES__
status() {
  printf '%s\n' "$1" >/tmp/devcore-v2-status.txt
  aws s3 cp /tmp/devcore-v2-status.txt "$RUN_URI/status.txt" --region us-east-1 --only-show-errors
}
finish() {
  rc=$?
  set +e
  if [ -n "$SYNC_PID" ]; then kill "$SYNC_PID" 2>/dev/null; fi
  timeout 240 aws s3 sync /opt/devcore-v2/results/ "$RUN_URI/results/" --region us-east-1 --only-show-errors
  timeout 60 aws s3 cp /var/log/devcore-v2.log "$RUN_URI/job.log" --region us-east-1 --only-show-errors
  if [ "$rc" -eq 0 ]; then status SUCCEEDED; else status "FAILED exit=$rc"; fi
  shutdown -h now
}
trap finish EXIT
status BOOTSTRAPPING
(while sleep 60; do
 aws s3 cp /var/log/devcore-v2.log "$RUN_URI/job.log" --region us-east-1 --only-show-errors >/dev/null 2>&1 || true
done) &
SYNC_PID=$!
dnf install -y python3.11 python3.11-pip libgomp
python3.11 -m venv /opt/devcore-v2/venv
aws s3 cp '__CODE__' /opt/devcore-v2/code.zip --region us-east-1 --only-show-errors
python3.11 -c "import zipfile; zipfile.ZipFile('/opt/devcore-v2/code.zip').extractall('/opt/devcore-v2/code')"
test -f /opt/devcore-v2/code/src/v2_cloud_smoke.py
test -f /opt/devcore-v2/code/requirements-v2.txt
/opt/devcore-v2/venv/bin/pip install --no-cache-dir -r /opt/devcore-v2/code/requirements-v2.txt
status SMOKE_TEST
/opt/devcore-v2/venv/bin/python /opt/devcore-v2/code/src/v2_cloud_smoke.py
status DOWNLOADING_DATA
aws s3 sync "$BUCKET_URI/input/train/" /opt/devcore-v2/data/train/ --region us-east-1 --only-show-errors
aws s3 sync "$CACHE_URI/" /opt/devcore-v2/index/ --region us-east-1 --only-show-errors
if [ '__MODE__' = 'probe' ]; then
  status RUNNING_PROBE
  timeout __SECONDS__ /opt/devcore-v2/venv/bin/python /opt/devcore-v2/code/src/v2_probe.py \
   --train-dir /opt/devcore-v2/data/train --index-dir /opt/devcore-v2/index \
   --output-dir /opt/devcore-v2/results --workers 8 --sample-modulus __MODULUS__
else
  mkdir -p /opt/devcore-v2/data/test
  aws s3 sync "$BUCKET_URI/input/test/" /opt/devcore-v2/data/test/ --region us-east-1 --only-show-errors
  status RUNNING_TRAIN_EVAL_AND_TEST
  timeout __SECONDS__ /opt/devcore-v2/venv/bin/python /opt/devcore-v2/code/src/v2_pipeline.py \
   --train-dir /opt/devcore-v2/data/train --test-dir /opt/devcore-v2/data/test \
   --index-dir /opt/devcore-v2/index --output-dir /opt/devcore-v2/results --workers 8
fi
status SAVING_INDEX_CACHE
timeout 480 aws s3 sync /opt/devcore-v2/index/ "$CACHE_URI/" --region us-east-1 --only-show-errors
'@
$bootstrap = $bootstrap.Replace('__RUN__',$runUri).Replace('__BUCKET__',$bucketUri).Replace('__CODE__',$codeUri).Replace('__CACHE__',$cacheUri).Replace('__MODE__',$Mode).Replace('__MINUTES__',[string]$lifetimeMinutes).Replace('__SECONDS__',[string]$jobSeconds).Replace('__MODULUS__',[string]$SampleModulus)
$bootstrapPath = Join-Path $stage 'bootstrap.sh'
[IO.File]::WriteAllText($bootstrapPath,($bootstrap -replace "`r`n","`n"),[Text.UTF8Encoding]::new($false))
if ($StageOnly) { Write-Output "Staged: $stage"; exit 0 }

$active = (Invoke-Aws @('ec2','describe-instances','--region',$region,'--filters','Name=instance-state-name,Values=running,pending','Name=tag:Name,Values=devcore-v2-*','--query','length(Reservations[].Instances[])','--output','text') | Out-String).Trim()
if ([int]$active -gt 0) { throw 'A DevCore v2 worker is already running. Inspect it before starting another.' }
$policyPath = Join-Path $stage 'role-policy.json'
$policy = (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'ec2-role-policy.json') -Raw).Replace('ACCOUNT',$account)
[IO.File]::WriteAllText($policyPath,$policy,[Text.UTF8Encoding]::new($false))
$null = Invoke-Aws @('iam','put-role-policy','--role-name','DevCoreEntityResolutionEc2Role','--policy-name','DevCoreEntityResolutionS3Access','--policy-document',"file://$policyPath")
$subnet = (Invoke-Aws @('ec2','describe-subnets','--region',$region,'--filters','Name=vpc-id,Values=vpc-04c41e12daa6b5411','Name=map-public-ip-on-launch,Values=true','--query',"Subnets[?AvailabilityZone=='us-east-1a'] | [0].SubnetId",'--output','text') | Out-String).Trim()
$group = (Invoke-Aws @('ec2','describe-security-groups','--region',$region,'--filters','Name=vpc-id,Values=vpc-04c41e12daa6b5411','Name=group-name,Values=default','--query','SecurityGroups[0].GroupId','--output','text') | Out-String).Trim()
$ami = (Invoke-Aws @('ssm','get-parameter','--region',$region,'--name','/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64','--query','Parameter.Value','--output','text') | Out-String).Trim()
if ($subnet -eq 'None' -or $group -eq 'None' -or $ami -eq 'None') { throw 'AWS launch prerequisites are missing' }
$blocks = Join-Path $stage 'blocks.json'
$volumeSize = if ($Mode -eq 'full') { 200 } else { 100 }
Write-Json $blocks @(@{DeviceName='/dev/xvda'; Ebs=@{VolumeSize=$volumeSize; VolumeType='gp3'; Encrypted=$true; DeleteOnTermination=$true}})
$tags = Join-Path $stage 'tags.json'
Write-Json $tags @(@{ResourceType='instance';Tags=@(@{Key='Name';Value="devcore-v2-$Mode-$label"},@{Key='Team';Value='DevCore'})},@{ResourceType='volume';Tags=@(@{Key='Team';Value='DevCore'})})
$instance = (Invoke-Aws @('ec2','run-instances','--region',$region,'--image-id',$ami,'--instance-type','r6i.2xlarge','--count','1','--subnet-id',$subnet,'--security-group-ids',$group,'--associate-public-ip-address','--iam-instance-profile','Name=DevCoreEntityResolutionEc2Profile','--user-data',"file://$bootstrapPath",'--block-device-mappings',"file://$blocks",'--tag-specifications',"file://$tags",'--metadata-options','HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=1','--instance-initiated-shutdown-behavior','terminate','--output','json') | Out-String) | ConvertFrom-Json
$run = @{instance_id=$instance.Instances[0].InstanceId; run_uri=$runUri; cache_uri=$cacheUri; mode=$Mode; max_hours=$MaxHours; started_utc=(Get-Date).ToUniversalTime().ToString('o'); poll_interval_minutes=15}
Write-Json (Join-Path $stage 'run.json') $run
Write-Json (Join-Path $workspace 'output\v2\latest-run.json') $run
$run | ConvertTo-Json
