# Download Qwen3.6-35B-A3B Q4_K_M and its MTP via individually verified 200MB GitHub assets.
# Rerunning skips previously SHA256-verified parts and already assembled original models.
param([string]$Directory = "C:\MODELS\Qwen3.6-35B-A3B")
$ErrorActionPreference = "Stop"

$repo = "toyro396133/personal-native-lm-5m"
$tag = "qwen3.6-35b-a3b-q4km-mtp-200mb"
$manifestName = "qwen3.6-35b-a3b-q4km-mtp.manifest.json"
New-Item -ItemType Directory -Path $Directory -Force | Out-Null

function Verify-Part([string]$path, $part) {
    if (!(Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $path).Length -ne [long]$part.bytes) { return $false }
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $part.sha256.ToLowerInvariant()
}

$manifestPath = Join-Path $Directory $manifestName
$received = $false
for ($attempt = 1; $attempt -le 5; $attempt++) {
    Write-Host "Downloading manifest (attempt $attempt)..." -ForegroundColor Cyan
    & gh release download $tag -R $repo -D $Directory -p $manifestName --clobber
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $manifestPath)) {
        $received = $true
        break
    }
    Start-Sleep -Seconds (3 * $attempt)
}
if (!$received) { throw "Manifest failed to download. Verify that GitHub Release is published." }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.format -ne "qwen36-200mb-parts-v1" -or @($manifest.files).Count -ne 2) {
    throw "Unexpected manifest format or missing files."
}
$expected = @("Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf", "mtp-Q4_K_M.gguf")
foreach ($model in $manifest.files) {
    if ($expected -notcontains $model.name -or $model.sha256 -notmatch '^[a-f0-9]{64}$' -or
        [long]$model.bytes -lt 1000000000 -or @($model.parts).Count -lt 5) {
        throw "Manifest model metadata invalid: $($model.name)"
    }
}

foreach ($model in $manifest.files) {
    $total = @($model.parts).Count
    $index = 0
    Write-Host "Starting $($model.name) ($total parts)..." -ForegroundColor Cyan
    foreach ($part in $model.parts) {
        $index++
        $path = Join-Path $Directory $part.name
        if (Verify-Part $path $part) {
            Write-Host "[$index/$total] Already verified: $($part.name)" -ForegroundColor Green
            continue
        }
        $downloaded = $false
        for ($attempt = 1; $attempt -le 5; $attempt++) {
            Write-Host "[$index/$total] Downloading $($part.name) attempt $attempt..." -ForegroundColor Cyan
            & gh release download $tag -R $repo -D $Directory -p $part.name --clobber
            if ($LASTEXITCODE -eq 0 -and (Verify-Part $path $part)) {
                $downloaded = $true
                break
            }
            Write-Warning "Part invalid or download interrupted. Retrying."
            Start-Sleep -Seconds (3 * $attempt)
        }
        if (!$downloaded) {
            throw "Failed part $($part.name). Rerun to skip every good part."
        }
    }
    $target = Join-Path $Directory $model.name
    if (Test-Path -LiteralPath $target) {
        if ((Get-Item -LiteralPath $target).Length -ne [long]$model.bytes -or
            (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $model.sha256.ToLowerInvariant()) {
            throw "Target GGUF exists but did not verify; keep it for inspection: $target"
        }
        Write-Host "Already assembled and verified: $target" -ForegroundColor Green
        continue
    }

    $assembling = "$target.assembling"
    Write-Host "Assembling $($model.name)..." -ForegroundColor Cyan
    $out = [System.IO.File]::Open($assembling, [System.IO.FileMode]::Create)
    try {
        foreach ($part in $model.parts) {
            $src = [System.IO.File]::OpenRead((Join-Path $Directory $part.name))
            try { $src.CopyTo($out) } finally { $src.Dispose() }
        }
    } finally { $out.Dispose() }
    $ok = (Get-Item -LiteralPath $assembling).Length -eq [long]$model.bytes -and
        (Get-FileHash -LiteralPath $assembling -Algorithm SHA256).Hash.ToLowerInvariant() -eq $model.sha256.ToLowerInvariant()
    if (!$ok) { throw "Failed full GGUF SHA256 validation, leaving assembling file for inspection." }
    Move-Item -LiteralPath $assembling -Destination $target
    Write-Host "VERIFIED ORIGINAL: $target" -ForegroundColor Green
}
Write-Host "SUCCESS! Both Qwen3.6 base model and Q4_K_M MTP verified." -ForegroundColor Green
Write-Host "Do not delete part files until both complete GGUF files are present and verified."
