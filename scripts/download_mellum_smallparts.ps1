# Download Mellum in 200MB chunks, reuse verified bytes from the old large partial files.
param([string]$Directory = "C:\MODELS\Mellum2.1")
$ErrorActionPreference = "Stop"
$repo = "toyro396133/personal-native-lm-5m"
$tag = "mellum2.1-12b-a2.5b-thinking-q4_k_m-smallparts"
$model = "Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf"
$manifestName = "$model.smallparts.manifest.json"
New-Item -ItemType Directory -Force -Path $Directory | Out-Null

Write-Host "Fetching verified 200MB release manifest..." -ForegroundColor Cyan
& gh release download $tag -R $repo -D $Directory -p $manifestName --clobber
if ($LASTEXITCODE -ne 0) { throw "Failed to download smallparts manifest." }
$manifest = Get-Content -LiteralPath (Join-Path $Directory $manifestName) -Raw | ConvertFrom-Json
if ($manifest.file -ne $model -or [long]$manifest.size_bytes -ne 8071295264 -or
    $manifest.sha256 -ne "ecc4d5b8107fc219e23c494b2d107552137637b6c6002ee713e0ba8af7fa2755") {
    throw "Unexpected release manifest, refusing to combine or download."
}

function Test-VerifiedPart($path, $part) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $path).Length -ne [long]$part.bytes) { return $false }
    $got = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    return $got -eq $part.sha256.ToLowerInvariant()
}

function Recover-FromLargePart($path, $part, $manifest, $dir, $model) {
    $originalSize = [long]$manifest.original_part_size_bytes
    $offset = [long]$part.offset
    $idx = [int][math]::Floor($offset / $originalSize) + 1
    $innerOffset = $offset % $originalSize
    $oldName = "$model.part{0:D2}" -f $idx
    $oldPath = Join-Path $dir $oldName
    if (!(Test-Path -LiteralPath $oldPath -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $oldPath).Length -lt ($innerOffset + [long]$part.bytes)) { return $false }
    $inputStream = [System.IO.File]::OpenRead($oldPath)
    try {
        [void]$inputStream.Seek($innerOffset, [System.IO.SeekOrigin]::Begin)
        $outputStream = [System.IO.File]::Create($path)
        try {
            $buffer = New-Object byte[] (4 * 1024 * 1024)
            $remaining = [long]$part.bytes
            while ($remaining -gt 0) {
                $wanted = [int][math]::Min($buffer.Length, $remaining)
                $read = $inputStream.Read($buffer, 0, $wanted)
                if ($read -le 0) { throw "Unexpected EOF in legacy partial file" }
                $outputStream.Write($buffer, 0, $read)
                $remaining -= $read
            }
        } finally {
            $outputStream.Dispose()
        }
    } finally {
        $inputStream.Dispose()
    }
    return (Test-VerifiedPart $path $part)
}

$total = @($manifest.parts).Count
$index = 0
foreach ($part in $manifest.parts) {
    $index++
    $path = Join-Path $Directory $part.name
    if (Test-VerifiedPart $path $part) {
        Write-Host "[$index/$total] Already verified: $($part.name)" -ForegroundColor Green
        continue
    }
    $recovered = $false
    try {
        $recovered = Recover-FromLargePart $path $part $manifest $Directory $model
    } catch {
        Write-Warning "Could not recover $($part.name) from previous partial: $_"
    }
    if ($recovered) {
        Write-Host "[$index/$total] Recovered from prior downloads!" -ForegroundColor Green
        continue
    }
    $good = $false
    for ($attempt = 1; $attempt -le 4; $attempt++) {
        Write-Host "[$index/$total] Downloading $($part.name), attempt $attempt..." -ForegroundColor Cyan
        & gh release download $tag -R $repo -D $Directory -p $part.name --clobber
        if ($LASTEXITCODE -eq 0 -and (Test-VerifiedPart $path $part)) {
            $good = $true
            break
        }
        Write-Warning "Incomplete or invalid part; retrying."
        Start-Sleep -Seconds (3 * $attempt)
    }
    if (!$good) { throw "Download failed: $($part.name). Rerun the script to resume verified chunks." }
}

$target = Join-Path $Directory $model
if (Test-Path -LiteralPath $target) {
    $full = Get-Item -LiteralPath $target
    if ($full.Length -eq [long]$manifest.size_bytes -and
        (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -eq
            $manifest.sha256.ToLowerInvariant()) {
        Write-Host "GGUF already verified: $target" -ForegroundColor Green
        exit 0
    }
    throw "An existing GGUF file is incomplete. Rename it manually before assembling."
}

Write-Host "Combining all validated chunks into one GGUF..." -ForegroundColor Cyan
$output = [System.IO.File]::Open($target, [System.IO.FileMode]::CreateNew)
try {
    foreach ($part in $manifest.parts) {
        $source = [System.IO.File]::OpenRead((Join-Path $Directory $part.name))
        try { $source.CopyTo($output) } finally { $source.Dispose() }
    }
} finally { $output.Dispose() }

if ((Get-Item -LiteralPath $target).Length -ne [long]$manifest.size_bytes -or
    (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne
        $manifest.sha256.ToLowerInvariant()) {
    throw "Combined GGUF failed size or SHA256 validation!"
}
Write-Host "SUCCESS: verified original GGUF at $target" -ForegroundColor Green
Write-Host "You can delete the verified smallparts after confirming SparkMoE loads the model."
