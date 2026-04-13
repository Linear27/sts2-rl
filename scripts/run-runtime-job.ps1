param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("validate", "run", "summary")]
    [string]$Command,
    [string]$Manifest = "",
    [string]$Config = "",
    [string]$JobName = "",
    [string]$OutputRoot = "",
    [string]$Source = "",
    [switch]$ReplaceExisting
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot

if ([string]::IsNullOrWhiteSpace($Manifest)) {
    $Manifest = Join-Path $repoRoot "configs\jobs\multi-instance.example.toml"
}
if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $repoRoot "configs\instances\local.example.toml"
}

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

switch ($Command) {
    "validate" {
        Invoke-Uv -Arguments @(
            "run",
            "sts2-rl",
            "instances",
            "job",
            "validate",
            "--manifest",
            $Manifest,
            "--config",
            $Config
        )
        break
    }
    "run" {
        $args = @(
            "run",
            "sts2-rl",
            "instances",
            "job",
            "run",
            "--manifest",
            $Manifest,
            "--config",
            $Config
        )
        if (-not [string]::IsNullOrWhiteSpace($JobName)) {
            $args += @("--job-name", $JobName)
        }
        if (-not [string]::IsNullOrWhiteSpace($OutputRoot)) {
            $args += @("--output-root", $OutputRoot)
        }
        if ($ReplaceExisting) {
            $args += "--replace-existing"
        }
        Invoke-Uv -Arguments $args
        break
    }
    "summary" {
        if ([string]::IsNullOrWhiteSpace($Source)) {
            throw "Source is required for the summary command."
        }
        Invoke-Uv -Arguments @(
            "run",
            "sts2-rl",
            "instances",
            "job",
            "summary",
            "--source",
            $Source
        )
        break
    }
}
