param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("start", "status", "collect", "plan", "train", "eval", "replay", "schedule", "normalize")]
    [string]$Command,
    [string]$SessionName = "",
    [string]$CheckpointPath = "",
    [string]$ResumeFrom = "",
    [int]$MaxStepsPerInstance = 0,
    [int]$MaxRunsPerInstance = 1,
    [int]$MaxCombatsPerInstance = 0,
    [int]$MaxEnvSteps = 0,
    [int]$MaxRuns = 1,
    [int]$MaxCombats = 0,
    [int]$MaxSessions = 3,
    [int]$RepeatCount = 3,
    [int]$CheckpointEveryRlSteps = 25,
    [int]$PrepareMaxSteps = 8,
    [int]$PrepareMaxIdlePolls = 40,
    [double]$RequestTimeoutSeconds = 30,
    [double]$TimeoutSeconds = 5,
    [string]$RenderingDriver = "opengl3",
    [string]$PrepareTarget = "",
    [string]$CheckpointSource = "latest",
    [int]$BestEvalRepeats = 3,
    [int]$BestEvalMaxEnvSteps = 0,
    [int]$BestEvalMaxRuns = 1,
    [int]$BestEvalMaxCombats = 0,
    [string]$BestEvalPrepareTarget = "main_menu",
    [int]$BestEvalPrepareMaxSteps = 8,
    [int]$BestEvalPrepareMaxIdlePolls = 40,
    [string]$BestEvalFallback = "latest",
    [int]$LaunchRetries = 1,
    [switch]$EnableDebugActions,
    [switch]$NoPrepareMainMenu
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$privateConfigPath = Join-Path $repoRoot "configs\instances\local.single.private.toml"
$exampleConfigPath = Join-Path $repoRoot "configs\instances\local.single.example.toml"
$configPath = if (Test-Path $privateConfigPath) { $privateConfigPath } else { $exampleConfigPath }
$outputRoot = Join-Path $repoRoot "data\trajectories"
$instanceRoot = Join-Path $repoRoot "runtime\inst-01"
$exePath = Join-Path $instanceRoot "SlayTheSpire2.exe"
$startScript = Join-Path $scriptRoot "start-sts2-instance.ps1"

function Invoke-Uv {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Push-Location $repoRoot
    try {
        & uv @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "uv command failed with exit code $LASTEXITCODE."
        }
    } finally {
        Pop-Location
    }
}

function Get-DefaultSessionName {
    return "single-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

switch ($Command) {
    "plan" {
        Invoke-Uv -Arguments @(
            "run",
            "sts2-rl",
            "instances",
            "plan",
            "--config",
            $configPath
        )
        break
    }
    "status" {
        Invoke-Uv -Arguments @(
            "run",
            "sts2-rl",
            "instances",
            "status",
            "--config",
            $configPath,
            "--timeout-seconds",
            "$TimeoutSeconds"
        )
        break
    }
    "collect" {
        if ([string]::IsNullOrWhiteSpace($SessionName)) {
            $SessionName = Get-DefaultSessionName
        }

        Invoke-Uv -Arguments @(
            "run",
            "sts2-rl",
            "collect",
            "rollouts",
            "--config",
            $configPath,
            "--output-root",
            $outputRoot,
            "--session-name",
            $SessionName,
            "--max-steps-per-instance",
            "$MaxStepsPerInstance",
            "--max-runs-per-instance",
            "$MaxRunsPerInstance",
            "--max-combats-per-instance",
            "$MaxCombatsPerInstance"
        )
        break
    }
    "train" {
        if ([string]::IsNullOrWhiteSpace($SessionName)) {
            $SessionName = "single-train-" + (Get-Date -Format "yyyyMMdd-HHmmss")
        }

        $args = @(
            "run",
            "sts2-rl",
            "train",
            "combat-dqn",
            "--base-url",
            "http://127.0.0.1:8080",
            "--output-root",
            (Join-Path $repoRoot "artifacts\training"),
            "--session-name",
            $SessionName,
            "--max-env-steps",
            "$MaxEnvSteps",
            "--max-runs",
            "$MaxRuns",
            "--max-combats",
            "$MaxCombats",
            "--checkpoint-every-rl-steps",
            "$CheckpointEveryRlSteps",
            "--request-timeout-seconds",
            "$RequestTimeoutSeconds"
        )
        if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) {
            $args += @("--resume-from", $ResumeFrom)
        }
        Invoke-Uv -Arguments $args
        break
    }
    "schedule" {
        if ([string]::IsNullOrWhiteSpace($SessionName)) {
            $SessionName = "single-schedule-" + (Get-Date -Format "yyyyMMdd-HHmmss")
        }

        $args = @(
            "run",
            "sts2-rl",
            "train",
            "combat-dqn-schedule",
            "--base-url",
            "http://127.0.0.1:8080",
            "--output-root",
            (Join-Path $repoRoot "artifacts\training-schedules"),
            "--schedule-name",
            $SessionName,
            "--max-sessions",
            "$MaxSessions",
            "--session-max-env-steps",
            "$MaxEnvSteps",
            "--session-max-runs",
            "$MaxRuns",
            "--session-max-combats",
            "$MaxCombats",
            "--checkpoint-source",
            $CheckpointSource,
            "--checkpoint-every-rl-steps",
            "$CheckpointEveryRlSteps",
            "--request-timeout-seconds",
            "$RequestTimeoutSeconds"
        )
        if ($CheckpointSource -eq "best_eval") {
            $args += @(
                "--best-eval-repeats",
                "$BestEvalRepeats",
                "--best-eval-max-env-steps",
                "$BestEvalMaxEnvSteps",
                "--best-eval-max-runs",
                "$BestEvalMaxRuns",
                "--best-eval-max-combats",
                "$BestEvalMaxCombats",
                "--best-eval-prepare-target",
                $BestEvalPrepareTarget,
                "--best-eval-prepare-max-steps",
                "$BestEvalPrepareMaxSteps",
                "--best-eval-prepare-max-idle-polls",
                "$BestEvalPrepareMaxIdlePolls",
                "--best-eval-fallback",
                $BestEvalFallback
            )
        }
        if (-not [string]::IsNullOrWhiteSpace($ResumeFrom)) {
            $args += @("--initial-resume-from", $ResumeFrom)
        }
        Invoke-Uv -Arguments $args
        break
    }
    "eval" {
        if ([string]::IsNullOrWhiteSpace($CheckpointPath)) {
            throw "CheckpointPath is required for the eval command."
        }
        if ([string]::IsNullOrWhiteSpace($SessionName)) {
            $SessionName = "single-eval-" + (Get-Date -Format "yyyyMMdd-HHmmss")
        }

        Invoke-Uv -Arguments @(
            "run",
            "sts2-rl",
            "eval",
            "combat-dqn",
            "--base-url",
            "http://127.0.0.1:8080",
            "--checkpoint-path",
            $CheckpointPath,
            "--output-root",
            (Join-Path $repoRoot "artifacts\eval"),
            "--session-name",
            $SessionName,
            "--max-env-steps",
            "$MaxEnvSteps",
            "--max-runs",
            "$MaxRuns",
            "--max-combats",
            "$MaxCombats",
            "--request-timeout-seconds",
            "$RequestTimeoutSeconds"
        )
        break
    }
    "replay" {
        if ([string]::IsNullOrWhiteSpace($CheckpointPath)) {
            throw "CheckpointPath is required for the replay command."
        }
        if ([string]::IsNullOrWhiteSpace($SessionName)) {
            $SessionName = "single-replay-" + (Get-Date -Format "yyyyMMdd-HHmmss")
        }

        $args = @(
            "run",
            "sts2-rl",
            "eval",
            "combat-dqn-replay",
            "--base-url",
            "http://127.0.0.1:8080",
            "--checkpoint-path",
            $CheckpointPath,
            "--output-root",
            (Join-Path $repoRoot "artifacts\replay"),
            "--suite-name",
            $SessionName,
            "--repeats",
            "$RepeatCount",
            "--max-env-steps",
            "$MaxEnvSteps",
            "--max-runs",
            "$MaxRuns",
            "--max-combats",
            "$MaxCombats",
            "--prepare-max-steps",
            "$PrepareMaxSteps",
            "--prepare-max-idle-polls",
            "$PrepareMaxIdlePolls",
            "--request-timeout-seconds",
            "$RequestTimeoutSeconds"
        )
        if (-not [string]::IsNullOrWhiteSpace($PrepareTarget)) {
            $args += @("--prepare-target", $PrepareTarget)
        } elseif ($NoPrepareMainMenu) {
            $args += "--no-prepare-main-menu"
        }
        Invoke-Uv -Arguments $args
        break
    }
    "normalize" {
        if ([string]::IsNullOrWhiteSpace($SessionName)) {
            $SessionName = "single-normalize-" + (Get-Date -Format "yyyyMMdd-HHmmss")
        }

        $target = $PrepareTarget
        if ([string]::IsNullOrWhiteSpace($target)) {
            $target = "main_menu"
        }

        Invoke-Uv -Arguments @(
            "run",
            "sts2-rl",
            "instances",
            "normalize",
            "--config",
            $configPath,
            "--output-root",
            (Join-Path $repoRoot "artifacts\runtime-normalize"),
            "--session-name",
            $SessionName,
            "--target",
            $target,
            "--max-steps",
            "$PrepareMaxSteps",
            "--max-idle-polls",
            "$PrepareMaxIdlePolls",
            "--request-timeout-seconds",
            "$RequestTimeoutSeconds"
        )
        break
    }
    "start" {
        if (-not (Test-Path -LiteralPath $exePath)) {
            throw "Provisioned single-instance executable not found at $exePath"
        }

        $startArgs = @{
            ExePath = $exePath
            ApiPort = 8080
            RenderingDriver = $RenderingDriver
            LaunchRetries = $LaunchRetries
        }

        if ($EnableDebugActions) {
            $startArgs["EnableDebugActions"] = $true
        }

        & $startScript @startArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Single-instance start script failed with exit code $LASTEXITCODE."
        }
        break
    }
}
