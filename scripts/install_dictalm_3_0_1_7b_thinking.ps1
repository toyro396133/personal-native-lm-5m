# Download and verify DictaLM 3.0 1.7B Thinking Q4_K_M from individual GitHub Release assets.
[CmdletBinding()]
param(
    [string]$ModelDirectory = (Join-Path $HOME "AI-Models\DictaLM-3.0-1.7B-Thinking")
)

$ErrorActionPreference = "Stop"
$Repo = "toyro396133/personal-native-lm-5m"
$Tag = "dictalm-3.0-1.7b-thinking-q4km-200mb"
$ManifestName = "dictalm-3.0-1.7b-thinking.manifest.json"
$ModelName = "DictaLM-3.0-1.7B-Thinking-Q4_K_M.gguf"
$ExpectedSha = "a92a6b58ea48a3ea1b6099bda070bb0dab440b17e2afc527874827be96d28c94"
$Temp = Join-Path $ModelDirectory ".download"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is required. Sign in first with: gh auth login"
}
New-Item -ItemType Directory -Force -Path $ModelDirectory, $Temp | Out-Null

function Get-Sha256([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Test-ExactFile([string]$Path, [long]$Size, [string]$Sha) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $item = Get-Item -LiteralPath $Path
    return ($item.Length -eq $Size -and (Get-Sha256 $Path) -ceq $Sha)
}
function Get-Asset([string]$Name, [long]$Size, [string]$Sha) {
    if ($Name -notmatch '^[A-Za-z0-9._-]+$') { throw "Unsafe asset name: $Name" }
    $path = Join-Path $Temp $Name
    if (Test-Path -LiteralPath $path) {
        if (Test-ExactFile $path $Size $Sha) { return $path }
        Remove-Item -LiteralPath $path -Force
    }
    Write-Host "Downloading $Name ..." -ForegroundColor Cyan
    & gh release download $Tag -R $Repo -D $Temp -p $Name --clobber
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Name" }
    if (-not (Test-ExactFile $path $Size $Sha)) { throw "Size/SHA-256 mismatch: $Name" }
    return $path
}

Write-Host "Downloading manifest..." -ForegroundColor Cyan
& gh release download $Tag -R $Repo -D $Temp -p $ManifestName --clobber
if ($LASTEXITCODE -ne 0) { throw "Could not download manifest." }
$manifest = Get-Content -LiteralPath (Join-Path $Temp $ManifestName) -Raw | ConvertFrom-Json
if ($manifest.format -cne "dictalm-3.0-1.7b-thinking-gguf-v1" -or
    $manifest.source_repo -cne "dicta-il/DictaLM-3.0-1.7B-Thinking-GGUF" -or
    $manifest.source_file -cne $ModelName -or
    $manifest.source_sha256 -cne $ExpectedSha -or
    $manifest.license -cne "Apache-2.0") {
    throw "Manifest provenance does not match the official DictaLM model."
}
$file = @($manifest.files | Where-Object { $_.path -ceq $ModelName })
if ($file.Count -ne 1 -or [long]$file[0].size_bytes -le 0 -or $file[0].sha256 -cne $ExpectedSha) {
    throw "Manifest model entry is missing or invalid."
}
$file = $file[0]
$parts = @($file.parts)
if ($parts.Count -lt 2) { throw "Unexpected chunk count." }
$sum = [long]0
foreach ($part in $parts) {
    if ($part.name -notmatch '^dictalm-3\.0-1\.7b-thinking-q4km\.part[0-9]{3}$' -or
        [long]$part.offset -ne $sum -or [long]$part.size_bytes -le 0 -or
        [long]$part.size_bytes -gt 200000000 -or $part.sha256 -notmatch '^[a-f0-9]{64}$') {
        throw "Invalid part metadata at offset $sum"
    }
    $sum += [long]$part.size_bytes
}
if ($sum -ne [long]$file.size_bytes) { throw "Part sizes do not match original model size." }

$target = Join-Path $ModelDirectory $ModelName
if (Test-Path -LiteralPath $target) {
    if (Test-ExactFile $target ([long]$file.size_bytes) $ExpectedSha) {
        Write-Host "MODEL ALREADY INSTALLED AND VERIFIED: $target" -ForegroundColor Green
        Remove-Item -LiteralPath $Temp -Recurse -Force -ErrorAction SilentlyContinue
        return
    }
    throw "A file already exists at the target path but does not match the official SHA-256. It was not overwritten: $target"
}
$drive = Get-PSDrive -Name $ModelDirectory.Substring(0,1) -ErrorAction Stop
$needed = [long]$file.size_bytes + 350000000
if ([long]$drive.Free -lt $needed) {
    throw ("Insufficient free space. Need approximately {0:N2} GB; available {1:N2} GB." -f ($needed/1GB), ($drive.Free/1GB))
}

$working = "$target.assembling"
if (Test-Path -LiteralPath $working) { Remove-Item -LiteralPath $working -Force }
$stream = [System.IO.File]::Open($working, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
try {
    $index = 0
    foreach ($part in $parts) {
        $index++
        $partPath = Get-Asset ([string]$part.name) ([long]$part.size_bytes) ([string]$part.sha256)
        $input = [System.IO.File]::OpenRead($partPath)
        try { $input.CopyTo($stream) } finally { $input.Dispose() }
        $stream.Flush()
        Remove-Item -LiteralPath $partPath -Force
        Write-Host ("[{0}/{1}] Verified and assembled." -f $index, $parts.Count) -ForegroundColor Green
    }
} finally {
    $stream.Dispose()
}
if (-not (Test-ExactFile $working ([long]$file.size_bytes) $ExpectedSha)) {
    throw "Final model SHA-256 verification failed. Assembly preserved at: $working"
}
Move-Item -LiteralPath $working -Destination $target
Remove-Item -LiteralPath $Temp -Recurse -Force -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "DictaLM 3.0 1.7B Thinking installed and SHA-256 verified!" -ForegroundColor Green
Write-Host "Model: $target"
Write-Host "No ZIP was downloaded."
