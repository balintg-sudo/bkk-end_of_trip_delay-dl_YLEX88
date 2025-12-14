$WorkDir = $PWD.Path

Write-Host "--- Starting project ---"
Write-Host "Work directory: $WorkDir"

# Ellenőrizzük, hogy létezik-e a 'data' mappa
if (-not (Test-Path "$WorkDir\data")) {
    Write-Host "ERROR: can't find 'data' directory in $WorkDir" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "$WorkDir\log")) {
    New-Item -ItemType Directory -Force -Path "$WorkDir\log" | Out-Null
}

Write-Host "Running docker container..."

docker run `
    --memory="12g" `
    --memory-swap="-1" `
    -v "$WorkDir\data:/app/data" `
    -v "$WorkDir\log:/app/log" `
    dl-project > "$WorkDir\log\run.log" 2>&1

Write-Host "Finished running. Log file at 'log/run.log" -ForegroundColor Green