param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [switch]$SkipDownload
)
$ErrorActionPreference = 'Stop'
$RawDir = Join-Path $ProjectRoot 'data\raw'
$DatasetDir = Join-Path $RawDir 'Task01_BrainTumour'
$Archive = Join-Path $RawDir 'Task01_BrainTumour.tar'
New-Item -ItemType Directory -Force -Path $RawDir | Out-Null
if (-not (Test-Path (Join-Path $DatasetDir 'dataset.json'))) {
    if (-not $SkipDownload -and -not (Test-Path $Archive)) {
        $Aws = (Get-Command aws -ErrorAction SilentlyContinue).Source
        if (-not $Aws -and (Test-Path 'C:\Program Files\Amazon\AWSCLIV2\aws.exe')) {
            $Aws = 'C:\Program Files\Amazon\AWSCLIV2\aws.exe'
        }
        if (-not $Aws) { throw 'AWS CLI was not found. Install AWS CLI v2 or place Task01_BrainTumour.tar in data\raw.' }
        Write-Host 'Downloading the public MSD Task01_BrainTumour archive (~7.1 GiB)...'
        & $Aws s3 cp s3://msd-for-monai/Task01_BrainTumour.tar $Archive --no-sign-request --region us-west-2
        if ($LASTEXITCODE -ne 0) { throw "AWS CLI download failed with exit code $LASTEXITCODE" }
    }
    if (-not (Test-Path $Archive)) { throw "Archive not found: $Archive" }
    Write-Host 'Extracting dataset archive...'
    tar -xf $Archive -C $RawDir
    if ($LASTEXITCODE -ne 0) { throw "tar extraction failed with exit code $LASTEXITCODE" }
}
foreach ($Required in @('dataset.json', 'imagesTr', 'labelsTr')) {
    if (-not (Test-Path (Join-Path $DatasetDir $Required))) { throw "Dataset extraction incomplete; missing $Required under $DatasetDir" }
}
$ImageCount = (Get-ChildItem (Join-Path $DatasetDir 'imagesTr') -File | Where-Object { $_.Name -match '\.nii(\.gz)?$' -and -not $_.Name.StartsWith('._') }).Count
$LabelCount = (Get-ChildItem (Join-Path $DatasetDir 'labelsTr') -File | Where-Object { $_.Name -match '\.nii(\.gz)?$' -and -not $_.Name.StartsWith('._') }).Count
if ($ImageCount -ne $LabelCount -or $ImageCount -lt 10) { throw "Image/label count mismatch or incomplete dataset: images=$ImageCount labels=$LabelCount" }
Write-Host "Dataset ready: $DatasetDir"
Write-Host "Labelled images: $ImageCount | Labels: $LabelCount"
