param(
    [string]$InstallDir = $env:TRAE_WORK_INSTALL_DIR,
    [string]$BindHost = $(if ($env:TRAEWORK_NATIVE_BRIDGE_HOST) { $env:TRAEWORK_NATIVE_BRIDGE_HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:TRAEWORK_NATIVE_BRIDGE_PORT) { [int]$env:TRAEWORK_NATIVE_BRIDGE_PORT } else { 40006 })
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = "E:\TRAE SOLO CN"
}

$moduleDir = Join-Path $InstallDir "resources\app\modules\ai-agent"
$required = @(
    (Join-Path $moduleDir "ai_agent.dll"),
    (Join-Path $moduleDir "sscronet.dll"),
    (Join-Path $moduleDir "meta.json"),
    (Join-Path $moduleDir "start.bat")
)
$missing = @($required | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missing.Count -gt 0) {
    throw "TraeWork ai-agent installation is incomplete: $($missing -join ', ')"
}

$env:TRAE_WORK_INSTALL_DIR = $InstallDir
$env:TRAEWORK_NATIVE_BRIDGE_HOST = $BindHost
$env:TRAEWORK_NATIVE_BRIDGE_PORT = [string]$Port
$env:TRAEWORK_NATIVE_ENABLED = "true"

Write-Host "TraeWork native bridge"
Write-Host ("  install: {0}" -f $InstallDir)
Write-Host ("  listen:  http://{0}:{1}" -f $BindHost, $Port)
Write-Host ("  AHA:     127.0.0.1:{0}" -f ($(if ($env:TRAEWORK_NATIVE_AHA_PORT) { $env:TRAEWORK_NATIVE_AHA_PORT } else { "40005" })))
Write-Host "No DLLs are copied; the helper uses the existing TraeWork installation."

python (Join-Path $PSScriptRoot "traework_native_helper.py") --host $BindHost --port $Port
