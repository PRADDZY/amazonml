param(
    [int]$IntervalSeconds = 900,
    [switch]$Once
)

$ErrorActionPreference = 'Continue'
$Project = 'amazonmlchallenge-509910'
$Zone = 'us-central1-a'
$Instance = 'devcore-v3-20260927'
$RemoteCommand = 'bash /home/daithankarpratik/app/status-v3.sh'

$Gcloud = Get-Command 'gcloud.cmd' -ErrorAction SilentlyContinue
if ($Gcloud) {
    $GcloudPath = $Gcloud.Source
} else {
    $GcloudPath = Join-Path $env:LOCALAPPDATA 'Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd'
}
if (-not (Test-Path -LiteralPath $GcloudPath)) {
    throw "gcloud.cmd not found: $GcloudPath"
}

do {
    Write-Host ''
    Write-Host ('=' * 78)
    Write-Host ('DEVCORE V3 | checked {0} | {1} / {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'), $Zone, $Instance)
    Write-Host ('=' * 78)

    $vm = & $GcloudPath compute instances describe $Instance --project=$Project --zone=$Zone --format='value(status,networkInterfaces[0].accessConfigs[0].natIP)' 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host ('VM status and external IP: ' + ($vm -join ' '))
        $worker = & $GcloudPath compute ssh $Instance --project=$Project --zone=$Zone --quiet --strict-host-key-checking=no --command $RemoteCommand 2>&1
        if ($LASTEXITCODE -eq 0) {
            $worker | ForEach-Object { Write-Host $_ }
        } else {
            Write-Host 'Worker details unavailable:'
            $worker | Select-Object -Last 8 | ForEach-Object { Write-Host $_ }
        }
    } else {
        Write-Host 'VM status unavailable:'
        $vm | ForEach-Object { Write-Host $_ }
    }

    if (-not $Once) {
        Write-Host ''
        Write-Host ('Next check in {0} minutes. Press Ctrl+C to stop.' -f [math]::Ceiling($IntervalSeconds / 60))
        Start-Sleep -Seconds $IntervalSeconds
    }
} while (-not $Once)
