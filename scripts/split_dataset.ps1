param(
    [string]$InputDir = "data\raw",
    [string]$OutputDir = "data",
    [double]$TrainRatio = 0.8,
    [int]$Seed = 42
)

if ($TrainRatio -le 0 -or $TrainRatio -ge 1) {
    throw "-TrainRatio must be between 0 and 1."
}

if (-not (Test-Path -LiteralPath $InputDir)) {
    throw "Input folder does not exist: $InputDir"
}

$images = Get-ChildItem -LiteralPath $InputDir -File |
    Where-Object { $_.Extension.ToLowerInvariant() -in @(".jpg", ".jpeg") }

if ($images.Count -eq 0) {
    throw "No JPG/JPEG images found in $InputDir."
}

$random = [System.Random]::new($Seed)
$shuffled = $images | Sort-Object { $random.Next() }
$splitIndex = [Math]::Floor($shuffled.Count * $TrainRatio)

$trainImages = @($shuffled | Select-Object -First $splitIndex)
$testImages = @($shuffled | Select-Object -Skip $splitIndex)

$trainDir = Join-Path $OutputDir "train\images"
$testDir = Join-Path $OutputDir "test\images"

New-Item -ItemType Directory -Force -Path $trainDir | Out-Null
New-Item -ItemType Directory -Force -Path $testDir | Out-Null

foreach ($image in $trainImages) {
    Copy-Item -LiteralPath $image.FullName -Destination (Join-Path $trainDir $image.Name) -Force
}

foreach ($image in $testImages) {
    Copy-Item -LiteralPath $image.FullName -Destination (Join-Path $testDir $image.Name) -Force
}

Write-Host "Total images: $($images.Count)"
Write-Host "Train images: $($trainImages.Count) -> $trainDir"
Write-Host "Test images: $($testImages.Count) -> $testDir"
