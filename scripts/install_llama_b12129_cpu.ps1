# Install the pinned llama.cpp b12129 Windows x64 CPU runtime from individual GitHub release assets.
[CmdletBinding()]
param(
    [string]$Directory = "C:\MODELS\llama.cpp-b12129-win-cpu-x64"
)

$ErrorActionPreference = "Stop"

$Repo = "toyro396133/personal-native-lm-5m"
$Tag = "llama-cpp-b12129-win-cpu-x64"
$ManifestName = "llama-cpp-b12129-win-cpu-x64.manifest.json"
$ExpectedArchiveSha256 = "dc0a41cba90e986652d18c41675c01d316d2df1232138421172be129cac416b5"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is required. Install it and sign in with: gh auth login"
}

New-Item -ItemType Directory -Force -Path $Directory | Out-Null
$ManifestPath = Join-Path $Directory $ManifestName

if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    Write-Host "Downloading verified file manifest..." -ForegroundColor Cyan
    & gh release download $Tag -R $Repo -D $Directory -p $ManifestName
    if ($LASTEXITCODE -ne 0) { throw "Could not download the runtime manifest from GitHub." }
}

$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if ($Manifest.format -cne "llama-cpp-individual-files-v1" -or
    $Manifest.source_repo -cne "ggml-org/llama.cpp" -or
    $Manifest.source_tag -cne "b12129" -or
    $Manifest.source_archive_sha256 -cne $ExpectedArchiveSha256 -or
    $Manifest.target -cne "Windows x64 CPU") {
    throw "Manifest metadata does not match the pinned llama.cpp b12129 CPU build."
}

$Assets = @($Manifest.assets)
if ($Assets.Count -lt 3) { throw "Manifest contains too few runtime assets." }

# Validate every manifest entry before downloading anything.
$Seen = @{}
foreach ($Asset in $Assets) {
    if ($Asset.name -notmatch '^[A-Za-z0-9._-]+$' -or
        [long]$Asset.size_bytes -le 0 -or
        $Asset.sha256 -notmatch '^[a-f0-9]{64}$' -or
        $Seen.ContainsKey($Asset.name)) {
        throw "Invalid or duplicate asset entry in manifest."
    }
    $Seen[$Asset.name] = $true
}

foreach ($Asset in $Assets) {
    $Path = Join-Path $Directory $Asset.name
    $Valid = $false

    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        $Item = Get-Item -LiteralPath $Path
        if ($Item.Length -eq [long]$Asset.size_bytes) {
            $Hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
            $Valid = ($Hash -eq $Asset.sha256)
        }
    }

    if (-not $Valid) {
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            Remove-Item -LiteralPath $Path -Force
        }
        Write-Host "Downloading $($Asset.name)..." -ForegroundColor Cyan
        & gh release download $Tag -R $Repo -D $Directory -p $Asset.name --clobber
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $($Asset.name)" }

        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "Downloaded asset is missing: $($Asset.name)"
        }
        $Item = Get-Item -LiteralPath $Path
        if ($Item.Length -ne [long]$Asset.size_bytes) {
            throw "File size mismatch: $($Asset.name)"
        }
        $Hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Hash -ne $Asset.sha256) {
            throw "SHA256 mismatch: $($Asset.name)"
        }
    }

    Write-Host ("VERIFIED  {0}  ({1:N0} bytes)" -f $Asset.name, [long]$Asset.size_bytes) -ForegroundColor Green
}

foreach ($Required in @("llama-server.exe", "llama-cli.exe", "start_lfm3.ps1")) {
    if (-not (Test-Path -LiteralPath (Join-Path $Directory $Required) -PathType Leaf)) {
        throw "Required runtime file is missing: $Required"
    }
}

Write-Host ""
Write-Host "Installation completed. Every downloaded runtime file passed SHA256 verification." -ForegroundColor Green
Write-Host "Runtime directory: $Directory"
Write-Host ('To start LFM3, run: powershell -NoProfile -ExecutionPolicy Bypass -File "' + $Directory + '\start_lfm3.ps1"') -ForegroundColor Cyan
