# Low-disk, interruption-resumable assembly for LiquidAI LFM3-1.7B Q4_K_M.
# Source parts are deleted ONLY after their exact output segments pass SHA256.
param(
    [string]$Directory = "C:\MODELS\LFM3-1.7B",
    [switch]$ReclaimVerifiedParts
)
$ErrorActionPreference = "Stop"
if (!$ReclaimVerifiedParts) {
    throw "This operation removes parts only after verifying their copies. Run with -ReclaimVerifiedParts to authorize this."
}

$repo = "toyro396133/personal-native-lm-5m"
$tag = "lfm3-1.7b-q4-k-m-200mb"
$manifestFile = "lfm3-1.7b-q4km.manifest.json"
$spec = @{
    name = "LFM3-1.7B-Q4_K_M.gguf"
    size = [long]1099627808
    sha = "3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4"
    count = 6
}
New-Item -ItemType Directory -Force -Path $Directory | Out-Null
$manifestPath = Join-Path $Directory $manifestFile
if (!(Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    & gh release download $tag -R $repo -D $Directory -p $manifestFile
    if ($LASTEXITCODE -ne 0) { throw "Unable to obtain manifest from GitHub." }
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.format -cne "lfm3-200mb-parts-v1" -or
    $manifest.source_repo -cne "LiquidAI/LFM3-1.7B-GGUF" -or
    $manifest.source_revision -cne "b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf" -or
    $manifest.file -cne $spec.name -or [long]$manifest.size_bytes -ne $spec.size -or
    $manifest.sha256 -cne $spec.sha) {
    throw "Manifest metadata does not match the pinned original GGUF."
}
$parts = @($manifest.parts)
if ($parts.Count -ne $spec.count) { throw "Unexpected number of parts in manifest." }

# Validate ALL names, offsets, sizes and hashes before touching any part.
$expectedOffset = [long]0
for ($i = 0; $i -lt $spec.count; $i++) {
    $part = $parts[$i]
    $expectedName = "{0}.part{1:D3}" -f $spec.name,($i + 1)
    if ($part.name -cne $expectedName -or [long]$part.offset -ne $expectedOffset -or
        [long]$part.bytes -le 0 -or [long]$part.bytes -gt 200000000 -or
        $part.sha256 -notmatch '^[a-f0-9]{64}$') {
        throw "Invalid manifest entry at part $($i + 1)."
    }
    $expectedOffset += [long]$part.bytes
}
if ($expectedOffset -ne $spec.size) { throw "Manifest part sizes do not match original GGUF." }

function Get-SegmentHash([string]$Path, [long]$Offset, [long]$Length) {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open,
                        [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    $hash = [System.Security.Cryptography.IncrementalHash]::CreateHash(
                [System.Security.Cryptography.HashAlgorithmName]::SHA256)
    try {
        if ($stream.Length -lt ($Offset + $Length)) { return "" }
        [void]$stream.Seek($Offset, [System.IO.SeekOrigin]::Begin)
        $buffer = New-Object byte[] (4 * 1024 * 1024)
        $remaining = $Length
        while ($remaining -gt 0) {
            $need = [int][Math]::Min([long]$buffer.Length, $remaining)
            $read = $stream.Read($buffer,0,$need)
            if ($read -le 0) { return "" }
            $hash.AppendData($buffer,0,$read)
            $remaining -= $read
        }
        return [BitConverter]::ToString($hash.GetHashAndReset()).Replace("-","").ToLowerInvariant()
    } finally {
        $hash.Dispose()
        $stream.Dispose()
    }
}

function Test-SourcePart([string]$Path, $Part) {
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -ne [long]$Part.bytes) { return $false }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $Part.sha256
}

function Ensure-SourcePart([string]$Path, $Part) {
    if (Test-SourcePart $Path $Part) { return }
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        Write-Host "Fetching missing/corrupt $($Part.name) ($attempt/5)" -ForegroundColor Yellow
        & gh release download $tag -R $repo -D $Directory -p $Part.name --clobber
        if ($LASTEXITCODE -eq 0 -and (Test-SourcePart $Path $Part)) { return }
        Start-Sleep -Seconds (3 * $attempt)
    }
    throw "Unable to obtain verified original part: $($Part.name)"
}

$target = Join-Path $Directory $spec.name
$working = "$target.assembling"
if (Test-Path -LiteralPath $target -PathType Leaf) {
    if ((Get-Item -LiteralPath $target).Length -eq $spec.size -and
        (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -eq $spec.sha) {
        Write-Host "COMPLETE: The original GGUF already exists and is SHA256 verified." -ForegroundColor Green
        exit 0
    }
    throw "An existing GGUF file is not verified. Nothing was changed: $target"
}

Write-Host "Resumable low-space assembly: $($spec.name)" -ForegroundColor Cyan
Write-Host "Each source part is deleted ONLY after the matching output segment is SHA256 verified." -ForegroundColor Yellow

$index = 0
foreach ($part in $parts) {
    $index++
    $offset = [long]$part.offset
    $size = [long]$part.bytes
    $end = $offset + $size
    $sourcePath = Join-Path $Directory $part.name
    $currentSize = if (Test-Path -LiteralPath $working -PathType Leaf) {
        [long](Get-Item -LiteralPath $working).Length
    } else { [long]0 }

    if ($currentSize -ge $end) {
        $segmentSha = Get-SegmentHash $working $offset $size
        if ($segmentSha -eq $part.sha256) {
            if (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
                Remove-Item -LiteralPath $sourcePath -Force
            }
            Write-Host "[$index/$($spec.count)] Previously written segment verified." -ForegroundColor Green
            continue
        }
    }
    if ($currentSize -lt $offset) {
        throw "Output file has a gap before part $index. No parts were deleted in this step."
    }

    Ensure-SourcePart $sourcePath $part
    $input = [System.IO.File]::OpenRead($sourcePath)
    try {
        $output = [System.IO.File]::Open($working, [System.IO.FileMode]::OpenOrCreate,
                     [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
        try {
            $output.SetLength($offset)
            [void]$output.Seek($offset,[System.IO.SeekOrigin]::Begin)
            $input.CopyTo($output)
            $output.Flush($true)
        } finally {
            $output.Dispose()
        }
    } finally {
        $input.Dispose()
    }

    if ((Get-Item -LiteralPath $working).Length -lt $end -or
        (Get-SegmentHash $working $offset $size) -ne $part.sha256) {
        throw "Output part $index did not verify. The source part was kept."
    }
    Remove-Item -LiteralPath $sourcePath -Force
    Write-Host "[$index/$($spec.count)] Copied, verified and reclaimed part." -ForegroundColor Green
}

if ((Get-Item -LiteralPath $working).Length -ne $spec.size) {
    throw "Assembled file size mismatch. The work file was preserved for recovery."
}
Write-Host "Running full-file SHA256 verification..." -ForegroundColor Cyan
$fullSha = (Get-FileHash -LiteralPath $working -Algorithm SHA256).Hash.ToLowerInvariant()
if ($fullSha -ne $spec.sha) {
    throw "Full SHA256 mismatch. The work file was preserved. Rerun to check each segment and recover."
}
Move-Item -LiteralPath $working -Destination $target -ErrorAction Stop
Write-Host "SUCCESS! Verified GGUF: $target" -ForegroundColor Green
Write-Host "No MTP sidecar is published for LFM3-1.7B." -ForegroundColor Cyan
