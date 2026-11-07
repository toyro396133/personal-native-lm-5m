# Download and assemble the selected ACE-Step 1.5 core checkpoints from individual GitHub Release assets.
# Existing files are never overwritten: verified files are skipped; mismatched files stop the installer.
[CmdletBinding()]
param(
    [string]$CheckpointsDirectory = "C:\Users\i5567\ACE-Music\ACE-Step-1.5\checkpoints"
)

$ErrorActionPreference = "Stop"

$Repo = "toyro396133/personal-native-lm-5m"
$Tag = "ace-step-1.5-core-200mb"
$ManifestName = "ace-step-1.5-core.manifest.json"
$ExpectedRevision = "19671f406d603126926c1b7e2adc169acbcade22"
$TempDirectory = Join-Path $CheckpointsDirectory ".ace-step-download"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is required. Install it and sign in with: gh auth login"
}

New-Item -ItemType Directory -Force -Path $CheckpointsDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $TempDirectory | Out-Null

Write-Host "Downloading ACE-Step manifest..." -ForegroundColor Cyan
& gh release download $Tag -R $Repo -D $TempDirectory -p $ManifestName --clobber
if ($LASTEXITCODE -ne 0) {
    throw "Could not download the ACE-Step manifest from GitHub."
}

$ManifestPath = Join-Path $TempDirectory $ManifestName
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json

if ($Manifest.format -cne "ace-step-1.5-core-files-v1" -or
    $Manifest.source_repo -cne "ACE-Step/Ace-Step1.5" -or
    $Manifest.source_revision -cne $ExpectedRevision -or
    $Manifest.license -cne "MIT" -or
    @($Manifest.excluded_components) -notcontains "acestep-5Hz-lm-1.7B") {
    throw "Manifest provenance does not match the pinned ACE-Step 1.5 core package."
}

$ExpectedPaths = @(
    "config.json",
    "acestep-v15-turbo/model.safetensors",
    "acestep-v15-turbo/config.json",
    "acestep-v15-turbo/configuration_acestep_v15.py",
    "acestep-v15-turbo/modeling_acestep_v15_turbo.py",
    "acestep-v15-turbo/silence_latent.pt",
    "vae/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "Qwen3-Embedding-0.6B/model.safetensors",
    "Qwen3-Embedding-0.6B/tokenizer.json",
    "Qwen3-Embedding-0.6B/vocab.json",
    "Qwen3-Embedding-0.6B/merges.txt",
    "Qwen3-Embedding-0.6B/tokenizer_config.json",
    "Qwen3-Embedding-0.6B/special_tokens_map.json",
    "Qwen3-Embedding-0.6B/added_tokens.json",
    "Qwen3-Embedding-0.6B/chat_template.jinja",
    "Qwen3-Embedding-0.6B/config.json",
    "ACE-STEP-MIT-LICENSE.txt"
)

$Files = @($Manifest.files)
if ($Files.Count -ne $ExpectedPaths.Count) {
    throw "Unexpected manifest file count: $($Files.Count); expected $($ExpectedPaths.Count)."
}

$FilesByPath = @{}
foreach ($File in $Files) {
    $Relative = [string]$File.path
    if ([string]::IsNullOrWhiteSpace($Relative) -or
        [System.IO.Path]::IsPathRooted($Relative) -or
        $Relative -match '(^|[\\/])\.\.([\\/]|$)') {
        throw "Unsafe target path in manifest: $Relative"
    }
    if ($FilesByPath.ContainsKey($Relative)) {
        throw "Duplicate target path in manifest: $Relative"
    }
    if ([long]$File.size_bytes -le 0 -or $File.sha256 -notmatch '^[a-f0-9]{64}$') {
        throw "Invalid size or SHA-256 in manifest: $Relative"
    }
    $FilesByPath[$Relative] = $File
}
foreach ($RequiredPath in $ExpectedPaths) {
    if (-not $FilesByPath.ContainsKey($RequiredPath)) {
        throw "Required file is missing from manifest: $RequiredPath"
    }
}
if (($Files | Where-Object { $_.path -like "acestep-5Hz-lm-1.7B/*" }).Count -gt 0) {
    throw "The optional 1.7B language model must not be included in this package."
}

# Pin the three large weight files against the official Hugging Face SHA-256 values.
$PinnedWeights = @{
    "acestep-v15-turbo/model.safetensors" = @{
        Size = [long]4787825604
        Sha = "3f6e0797fad420a39bd33979eb6e840e30989e34a3794e843d23b60ec6e422d7"
    }
    "vae/diffusion_pytorch_model.safetensors" = @{
        Size = [long]337431388
        Sha = "da17edb604c40deaf09e9b24974e590d1ca83a374070e5d0884cfa4bed9a99b0"
    }
    "Qwen3-Embedding-0.6B/model.safetensors" = @{
        Size = [long]1191586416
        Sha = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
    }
}
foreach ($Path in $PinnedWeights.Keys) {
    $Entry = $FilesByPath[$Path]
    if ([long]$Entry.size_bytes -ne $PinnedWeights[$Path].Size -or
        $Entry.sha256 -cne $PinnedWeights[$Path].Sha) {
        throw "Pinned upstream weight metadata mismatch: $Path"
    }
}

function Test-FileMatch([string]$Path, [long]$ExpectedSize, [string]$ExpectedSha) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $Item = Get-Item -LiteralPath $Path
    if ($Item.Length -ne $ExpectedSize) { return $false }
    $Hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    return ($Hash -ceq $ExpectedSha)
}

function Get-VerifiedAsset([string]$AssetName, [long]$ExpectedSize, [string]$ExpectedSha) {
    if ($AssetName -notmatch '^[A-Za-z0-9._-]+$') {
        throw "Unsafe Release asset name: $AssetName"
    }
    $LocalPath = Join-Path $TempDirectory $AssetName
    if (Test-Path -LiteralPath $LocalPath -PathType Leaf) {
        if (Test-FileMatch $LocalPath $ExpectedSize $ExpectedSha) {
            return $LocalPath
        }
        Remove-Item -LiteralPath $LocalPath -Force
    }

    Write-Host "Downloading $AssetName..." -ForegroundColor Cyan
    & gh release download $Tag -R $Repo -D $TempDirectory -p $AssetName --clobber
    if ($LASTEXITCODE -ne 0) {
        throw "GitHub asset download failed: $AssetName"
    }
    if (-not (Test-FileMatch $LocalPath $ExpectedSize $ExpectedSha)) {
        throw "Downloaded asset failed size/SHA-256 verification: $AssetName"
    }
    return $LocalPath
}

function Get-SegmentHash([string]$Path, [long]$Offset, [long]$Length) {
    $Stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::Read
    )
    $Hash = [System.Security.Cryptography.IncrementalHash]::CreateHash(
        [System.Security.Cryptography.HashAlgorithmName]::SHA256
    )
    try {
        if ($Stream.Length -lt ($Offset + $Length)) { return "" }
        [void]$Stream.Seek($Offset, [System.IO.SeekOrigin]::Begin)
        $Buffer = New-Object byte[] (4 * 1024 * 1024)
        $Remaining = $Length
        while ($Remaining -gt 0) {
            $Need = [int][Math]::Min([long]$Buffer.Length, $Remaining)
            $Read = $Stream.Read($Buffer, 0, $Need)
            if ($Read -le 0) { return "" }
            $Hash.AppendData($Buffer, 0, $Read)
            $Remaining -= $Read
        }
        return [BitConverter]::ToString($Hash.GetHashAndReset()).Replace("-", "").ToLowerInvariant()
    } finally {
        $Hash.Dispose()
        $Stream.Dispose()
    }
}

# Check all existing files first. Nothing is overwritten if a file exists but is not an exact match.
$MissingBytes = [long]0
foreach ($File in $Files) {
    $Target = Join-Path $CheckpointsDirectory ([string]$File.path).Replace("/", "\")
    if (Test-Path -LiteralPath $Target -PathType Leaf) {
        if (-not (Test-FileMatch $Target ([long]$File.size_bytes) ([string]$File.sha256))) {
            throw "An existing file is not verified. It was left untouched: $Target"
        }
        Write-Host "ALREADY VERIFIED: $($File.path)" -ForegroundColor Green
    } else {
        $MissingBytes += [long]$File.size_bytes
    }
}

$DriveName = $CheckpointsDirectory.Substring(0, 1)
$Drive = Get-PSDrive -Name $DriveName -ErrorAction Stop
$SafetyMargin = [long]300000000
if ([long]$Drive.Free -lt ($MissingBytes + $SafetyMargin)) {
    throw ("Not enough free space on {0}: need about {1:N2} GB for missing files plus a safety margin; available {2:N2} GB." -f
        $DriveName, (($MissingBytes + $SafetyMargin) / 1GB), ($Drive.Free / 1GB))
}

Write-Host ""
Write-Host ("Files to install: {0}; remaining model data: {1:N2} GB" -f
    (($Files | Where-Object {
        $Target = Join-Path $CheckpointsDirectory ([string]$_.path).Replace("/", "\")
        -not (Test-Path -LiteralPath $Target -PathType Leaf)
    }).Count), ($MissingBytes / 1GB)) -ForegroundColor Cyan

foreach ($File in $Files) {
    $Relative = [string]$File.path
    $Target = Join-Path $CheckpointsDirectory $Relative.Replace("/", "\")
    $Parent = Split-Path -Parent $Target
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null

    if (Test-FileMatch $Target ([long]$File.size_bytes) ([string]$File.sha256)) {
        continue
    }

    $Parts = @($File.parts)
    if ($Parts.Count -eq 0) {
        $LocalAsset = Get-VerifiedAsset ([string]$File.asset_name) ([long]$File.size_bytes) ([string]$File.sha256)
        Move-Item -LiteralPath $LocalAsset -Destination $Target
        Write-Host "INSTALLED AND VERIFIED: $Relative" -ForegroundColor Green
        continue
    }

    $ExpectedOffset = [long]0
    foreach ($Part in $Parts) {
        if ([long]$Part.offset -ne $ExpectedOffset -or
            [long]$Part.size_bytes -le 0 -or
            [long]$Part.size_bytes -gt [long]200000000 -or
            $Part.sha256 -notmatch '^[a-f0-9]{64}$') {
            throw "Invalid chunk metadata for $Relative at offset $ExpectedOffset"
        }
        $ExpectedOffset += [long]$Part.size_bytes
    }
    if ($ExpectedOffset -ne [long]$File.size_bytes) {
        throw "Chunk sizes do not add up to the original file size: $Relative"
    }

    $Working = "$Target.assembling"
    Write-Host ""
    Write-Host "Assembling $Relative from $($Parts.Count) verified parts..." -ForegroundColor Cyan

    $PartIndex = 0
    foreach ($Part in $Parts) {
        $PartIndex++
        $Offset = [long]$Part.offset
        $Length = [long]$Part.size_bytes
        $End = $Offset + $Length
        $CurrentSize = if (Test-Path -LiteralPath $Working -PathType Leaf) {
            [long](Get-Item -LiteralPath $Working).Length
        } else {
            [long]0
        }

        if ($CurrentSize -ge $End) {
            $ExistingSegmentSha = Get-SegmentHash $Working $Offset $Length
            if ($ExistingSegmentSha -ceq $Part.sha256) {
                $StalePart = Join-Path $TempDirectory ([string]$Part.name)
                if (Test-Path -LiteralPath $StalePart -PathType Leaf) {
                    Remove-Item -LiteralPath $StalePart -Force
                }
                Write-Host ("[{0}/{1}] Segment already verified." -f $PartIndex, $Parts.Count) -ForegroundColor Green
                continue
            }
        }

        if ($CurrentSize -lt $Offset) {
            throw "The assembly file has a gap before part $PartIndex of $Relative. It was preserved."
        }

        $LocalPart = Get-VerifiedAsset ([string]$Part.name) $Length ([string]$Part.sha256)

        $InputStream = [System.IO.File]::OpenRead($LocalPart)
        try {
            $OutputStream = [System.IO.File]::Open(
                $Working,
                [System.IO.FileMode]::OpenOrCreate,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::None
            )
            try {
                $OutputStream.SetLength($Offset)
                [void]$OutputStream.Seek($Offset, [System.IO.SeekOrigin]::Begin)
                $InputStream.CopyTo($OutputStream)
                $OutputStream.Flush($true)
            } finally {
                $OutputStream.Dispose()
            }
        } finally {
            $InputStream.Dispose()
        }

        if ((Get-SegmentHash $Working $Offset $Length) -cne $Part.sha256) {
            throw "Output segment verification failed. The downloaded part was kept: $($Part.name)"
        }
        Remove-Item -LiteralPath $LocalPart -Force
        Write-Host ("[{0}/{1}] Copied, SHA-256 verified, temporary part reclaimed." -f
            $PartIndex, $Parts.Count) -ForegroundColor Green
    }

    if (-not (Test-FileMatch $Working ([long]$File.size_bytes) ([string]$File.sha256))) {
        throw "Final size/SHA-256 verification failed. The assembly file was preserved: $Working"
    }

    Move-Item -LiteralPath $Working -Destination $Target
    Write-Host "INSTALLED AND VERIFIED: $Relative" -ForegroundColor Green
}

# Final pass over every expected file.
foreach ($File in $Files) {
    $Target = Join-Path $CheckpointsDirectory ([string]$File.path).Replace("/", "\")
    if (-not (Test-FileMatch $Target ([long]$File.size_bytes) ([string]$File.sha256))) {
        throw "Final verification failed: $Target"
    }
}
Remove-Item -LiteralPath $TempDirectory -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "ACE-Step 1.5 core checkpoints installed and SHA-256 verified." -ForegroundColor Green
Write-Host "Checkpoints: $CheckpointsDirectory"
Write-Host "Included: Turbo + VAE + Qwen3-Embedding-0.6B + root config."
Write-Host "Excluded: acestep-5Hz-lm-1.7B."
Write-Host "No ZIP was downloaded to this PC."
