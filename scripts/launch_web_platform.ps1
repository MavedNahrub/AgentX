# AgentX v2.0 — Production Web Platform Launcher (FastAPI :8000 + Next.js :3000)
# Usage: .\scripts\launch_web_platform.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " Starting AgentX v2.0 Platform: FastAPI Quant Server (:8000) + Next.js (:3000)  " -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

# 0. Cleanly stop any stale FastAPI (:8000) or Next.js (:3000) listeners from previous runs
foreach ($Port in @(8000, 3000)) {
    $Conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($C in $Conns) {
        if ($C.OwningProcess -and $C.OwningProcess -gt 4) {
            Write-Host " -> Stopping previous listener on port $Port (PID $($C.OwningProcess))..." -ForegroundColor Yellow
            Stop-Process -Id $C.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}
Start-Sleep -Milliseconds 600

# 1. Start FastAPI Quant Telemetry + WebSocket Server in a dedicated PowerShell window (with --reload)
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$RepoRoot'; `$env:PYTHONPATH='.'; .\.venv\Scripts\python.exe -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000 --reload"
)

# 2. Run Next.js TradingView Frontend in the current terminal on http://localhost:3000
Set-Location "$RepoRoot\web"
npm.cmd run start
