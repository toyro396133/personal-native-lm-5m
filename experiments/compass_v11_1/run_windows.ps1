param(
    [Parameter(Mandatory=$true)][string]$MessagesPath,
    [Parameter(Mandatory=$true)][string]$ProjectId,
    [string]$OutputDir = "$env:USERPROFILE\Documents\CompassPrivate",
    [string]$ReviewsPath = "",
    [switch]$Personal,
    [switch]$OpenReview
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$messages = (Resolve-Path -LiteralPath $MessagesPath).Path
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$output = (Resolve-Path -LiteralPath $OutputDir).Path
$events = Join-Path $output "compass-v111-private-chat-events.jsonl"
$audit = Join-Path $output "compass-v111-ingest-audit.json"
$reviewHtml = Join-Path $output "compass-v111-private-offline-review.html"
$report = Join-Path $output "compass-v111-private-report.json"
$state = Join-Path $output "compass-v111-private-learned-state.json"
$scope = if ($Personal) { "personal" } else { "project" }
$id = if ($Personal) { "" } else { $ProjectId }

& python "$root\chat_ingest.py" --messages $messages --project $id --scope $scope --events $events --audit $audit
if ($LASTEXITCODE -ne 0) { throw "Chat import failed. No model data was written." }
& python "$root\offline_review.py" --events $events --html $reviewHtml
if ($LASTEXITCODE -ne 0) { throw "Offline review creation failed." }
Write-Host ""
Write-Host "Local-only private review page: $reviewHtml"
Write-Host "Nothing was sent to external services."
Write-Host "Open the file in a browser, confirm only corrections you personally checked, and export the JSON review file."
if ($OpenReview) { Start-Process -FilePath $reviewHtml }

$arguments = @(
    "$root\freeform_feedback.py",
    "--events", $events,
    "--report", $report,
    "--state", $state
)
if ($ReviewsPath -ne "") {
    $reviewFile = (Resolve-Path -LiteralPath $ReviewsPath).Path
    $arguments += @("--reviews", $reviewFile)
    Write-Host "Using explicit local review file: $reviewFile"
} else {
    Write-Host "No review supplied: ambiguous events stay unresolved."
}
& python @arguments
if ($LASTEXITCODE -ne 0) { throw "COMPASS replay failed. Check the original source or review binding." }
Write-Host "Experimental report: $report"
Write-Host "Local learned-state checkpoint: $state"
Write-Host "NOTE: This is a limited lexical online learner, not verified semantic understanding or approved policy."
