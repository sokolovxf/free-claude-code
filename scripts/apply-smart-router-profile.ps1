[CmdletBinding()]
param(
    [string] $ProfilePath = (Join-Path $PSScriptRoot "..\\.env.smart-router.example"),
    [string] $ConfigPath = (Join-Path $env:USERPROFILE ".fcc\\.env")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$managedKeys = @(
    "FCC_CONFIG_SCHEMA",
    "PORT",
    "MODEL",
    "MODEL_FALLBACKS",
    "FCC_VERIFIED_FREE_MODELS"
)

$ProfilePath = (Resolve-Path -LiteralPath $ProfilePath).Path
if (-not (Test-Path -LiteralPath $ProfilePath -PathType Leaf)) {
    throw "SmartRouter profile was not found: $ProfilePath"
}

$profileValues = @{}
foreach ($line in Get-Content -LiteralPath $ProfilePath) {
    if ($line -match "^([A-Z0-9_]+)=(.*)$" -and $managedKeys -contains $Matches[1]) {
        $profileValues[$Matches[1]] = $line
    }
}

foreach ($key in $managedKeys) {
    if (-not $profileValues.ContainsKey($key)) {
        throw "SmartRouter profile is missing required key: $key"
    }
}

$configDirectory = Split-Path -Parent $ConfigPath
New-Item -ItemType Directory -Force -Path $configDirectory | Out-Null

$backupPath = $null
$existingLines = @()
if (Test-Path -LiteralPath $ConfigPath -PathType Leaf) {
    $backupPath = "$ConfigPath.before-smart-router-$(Get-Date -Format yyyyMMddHHmmss).bak"
    Copy-Item -LiteralPath $ConfigPath -Destination $backupPath
    $existingLines = @(Get-Content -LiteralPath $ConfigPath)
}

$output = [Collections.Generic.List[string]]::new()
$written = @{}
foreach ($line in $existingLines) {
    if ($line -match "^([A-Z0-9_]+)=") {
        $key = $Matches[1]
        if ($managedKeys -contains $key) {
            if (-not $written.ContainsKey($key)) {
                $output.Add($profileValues[$key])
                $written[$key] = $true
            }
            continue
        }
    }
    $output.Add($line)
}

foreach ($key in $managedKeys) {
    if (-not $written.ContainsKey($key)) {
        $output.Add($profileValues[$key])
    }
}

Set-Content -LiteralPath $ConfigPath -Value $output -Encoding utf8
Write-Host "Applied SmartRouter profile to $ConfigPath"
if ($backupPath) {
    Write-Host "Previous config backed up at $backupPath"
}
