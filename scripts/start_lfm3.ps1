# Start LiquidAI LFM3-1.7B with the local llama.cpp CPU server.
[CmdletBinding()]
param(
    [string]$ModelPath = "C:\MODELS\LFM3-1.7B\LFM3-1.7B-Q4_K_M.gguf",
    [int]$Context = 4096,
    [int]$Threads = 2,
    [int]$Port = 9931
)

$ErrorActionPreference = "Stop"
$Server = Join-Path $PSScriptRoot "llama-server.exe"

if (-not (Test-Path -LiteralPath $Server -PathType Leaf)) {
    throw "llama-server.exe is missing. Run install_llama_b12129_cpu.ps1 first."
}
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw "Model file not found: $ModelPath"
}
if ($Context -lt 512 -or $Threads -lt 1 -or $Port -lt 1 -or $Port -gt 65535) {
    throw "Invalid context, thread count, or port."
}

Write-Host "Starting LFM3-1.7B on CPU..." -ForegroundColor Cyan
Write-Host "Model: $ModelPath"
Write-Host "Context: $Context | Threads: $Threads | Port: $Port"
Write-Host "When the server is ready, open http://127.0.0.1:$Port"
Write-Host "Press Ctrl+C here to stop the model." -ForegroundColor Yellow

& $Server -m $ModelPath -c $Context -t $Threads -ngl 0 --host 127.0.0.1 --port $Port --temp 0.1 --top-k 50 --repeat-penalty 1.1

if ($LASTEXITCODE -ne 0) {
    throw "llama-server exited with code $LASTEXITCODE"
}
