param(
    [Parameter(Mandatory = $true)]
    [string]$ExePath,
    [int]$LaunchRetries = 1,
    [int]$Attempts = 40,
    [int]$DelaySeconds = 2,
    [int]$ApiPort = 8080,
    [int]$SteamAppId = 2868840,
    [string]$RenderingDriver = "opengl3",
    [switch]$EnableDebugActions,
    [switch]$KeepExistingProcesses
)

$ErrorActionPreference = "Stop"

function Wait-ForHealth {
    param(
        [int]$MaxAttempts,
        [int]$SleepSeconds,
        [System.Diagnostics.Process]$Process,
        [string]$BaseUrl
    )

    for ($i = 0; $i -lt $MaxAttempts; $i++) {
        Start-Sleep -Seconds $SleepSeconds

        try {
            $response = Invoke-WebRequest -Uri ($BaseUrl.TrimEnd("/") + "/health") -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                return
            }
        } catch {
        }

        if ($Process.HasExited) {
            throw "Game process exited before /health became ready."
        }
    }

    throw "Timed out waiting for /health."
}

function Wait-ForPortRelease {
    param(
        [int]$MaxAttempts,
        [int]$SleepSeconds,
        [int]$Port
    )

    for ($i = 0; $i -lt $MaxAttempts; $i++) {
        try {
            $listenerActive = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop).Count -gt 0
        } catch {
            $listenerActive = $false
        }

        if (-not $listenerActive) {
            return
        }

        Start-Sleep -Seconds $SleepSeconds
    }
}

function Quote-Argument {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return '""'
    }

    if ($Value.Contains('"')) {
        $Value = $Value.Replace('"', '\"')
    }

    if ($Value.Contains(' ')) {
        return '"' + $Value + '"'
    }

    return $Value
}

$resolvedExePath = (Resolve-Path -LiteralPath $ExePath).Path
$gameRoot = Split-Path -Path $resolvedExePath -Parent
$appIdFile = Join-Path $gameRoot "steam_appid.txt"
$baseUrl = "http://127.0.0.1:$ApiPort"

# Local copied instances still need the Steam app id so Steamworks can bind correctly.
if (-not (Test-Path -LiteralPath $appIdFile)) {
    Set-Content -LiteralPath $appIdFile -Value ([string]$SteamAppId) -Encoding ascii -NoNewline
} else {
    $existingAppId = (Get-Content -LiteralPath $appIdFile -Raw).Trim()
    if ($existingAppId -ne [string]$SteamAppId) {
        Set-Content -LiteralPath $appIdFile -Value ([string]$SteamAppId) -Encoding ascii -NoNewline
    }
}

if (-not $KeepExistingProcesses) {
    $existing = Get-Process -Name "SlayTheSpire2" -ErrorAction SilentlyContinue
    if ($existing) {
        Stop-Process -Id $existing.Id -Force
        Start-Sleep -Seconds 2
        Wait-ForPortRelease -MaxAttempts 10 -SleepSeconds 1 -Port $ApiPort
    }
}

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $resolvedExePath
$startInfo.WorkingDirectory = $gameRoot
$startInfo.UseShellExecute = $false
$startInfo.EnvironmentVariables["STS2_API_PORT"] = [string]$ApiPort
$startInfo.EnvironmentVariables["SteamAppId"] = [string]$SteamAppId
$startInfo.EnvironmentVariables["SteamGameId"] = [string]$SteamAppId

if ($EnableDebugActions) {
    $startInfo.EnvironmentVariables["STS2_ENABLE_DEBUG_ACTIONS"] = "1"
} else {
    $startInfo.EnvironmentVariables.Remove("STS2_ENABLE_DEBUG_ACTIONS")
}

$arguments = @()
if ($RenderingDriver -and $RenderingDriver -ne "default") {
    $arguments += "--rendering-driver"
    $arguments += $RenderingDriver
}

if ($arguments.Count -gt 0) {
    $startInfo.Arguments = ($arguments | ForEach-Object { Quote-Argument $_ }) -join " "
}

$proc = $null
$lastFailure = $null

for ($launchAttempt = 0; $launchAttempt -le $LaunchRetries; $launchAttempt++) {
    try {
        $proc = [System.Diagnostics.Process]::Start($startInfo)
        Wait-ForHealth -MaxAttempts $Attempts -SleepSeconds $DelaySeconds -Process $proc -BaseUrl $baseUrl
        $lastFailure = $null
        break
    } catch {
        $lastFailure = $_

        if ($launchAttempt -ge $LaunchRetries) {
            throw
        }

        Start-Sleep -Seconds 2
    }
}

if ($null -ne $lastFailure) {
    throw $lastFailure
}

[pscustomobject]@{
    pid = $proc.Id
    api_port = $ApiPort
    base_url = $baseUrl
    exe_path = $resolvedExePath
    launch_retries = $LaunchRetries
    rendering_driver = $RenderingDriver
    steam_app_id = $SteamAppId
    debug_actions_enabled = [bool]$EnableDebugActions
    keep_existing_processes = [bool]$KeepExistingProcesses
    health = "ready"
} | ConvertTo-Json -Compress
