[CmdletBinding()]
param(
    [string]$PythonCommand = "python",
    [string]$VenvPath = ".venv",
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentPath = if ([IO.Path]::IsPathRooted($VenvPath)) {
    [IO.Path]::GetFullPath($VenvPath)
} else {
    [IO.Path]::GetFullPath((Join-Path $ProjectRoot $VenvPath))
}
$EnvironmentPython = Join-Path $EnvironmentPath "Scripts\python.exe"

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

function Confirm-PinnedRepository {
    param([string]$Name, [string]$Url, [string]$Commit)
    $Destination = Join-Path $ProjectRoot "vendor\$Name"
    if (-not (Test-Path -LiteralPath $Destination)) {
        if ($CheckOnly) { throw "Missing repository: $Destination" }
        $null = New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "vendor")
        Invoke-Checked "git" @("clone", "--no-checkout", "--depth", "1", "--", $Url, $Destination)
        Invoke-Checked "git" @("-C", $Destination, "fetch", "--depth", "1", "origin", $Commit)
        Invoke-Checked "git" @("-C", $Destination, "checkout", "--detach", $Commit)
    }
    # Existing repositories are never reset, checked out, cleaned, or overwritten.
    $Head = & git -C $Destination rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw "Not a readable Git checkout: $Destination" }
    if ($Head.Trim() -ne $Commit) {
        throw "$Name is at $Head; expected $Commit. Existing checkout was preserved."
    }
    $Dirty = & git -C $Destination status --porcelain
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect $Name working tree" }
    if ($Dirty) { throw "$Name contains local changes. They were preserved; use a separate clean project checkout." }
    Write-Host "Verified $Name $Commit"
}

Push-Location -LiteralPath $ProjectRoot
try {
    $null = Get-Command git -ErrorAction Stop
    if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
        if ($CheckOnly) { throw "Missing Python environment: $EnvironmentPython" }
        if (Test-Path -LiteralPath $EnvironmentPath) {
            throw "Environment path exists without Scripts/python.exe; refusing to overwrite: $EnvironmentPath"
        }
        $null = Get-Command $PythonCommand -ErrorAction Stop
        Invoke-Checked $PythonCommand @("-c", "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ required'")
        # Deliberately omit --system-site-packages for an isolated new environment.
        Invoke-Checked $PythonCommand @("-m", "venv", $EnvironmentPath)
    }
    $EnvironmentConfig = Join-Path $EnvironmentPath "pyvenv.cfg"
    if (-not (Test-Path -LiteralPath $EnvironmentConfig)) {
        throw "Missing pyvenv.cfg; refusing to install into an unverified environment"
    }
    if (Select-String -LiteralPath $EnvironmentConfig -Pattern '^include-system-site-packages\s*=\s*true' -Quiet) {
        Write-Warning "Existing environment inherits system packages. Preserving it; use -VenvPath .venv-clean for an isolated reproduction."
    }

    Confirm-PinnedRepository "FinRL" "https://github.com/AI4Finance-Foundation/FinRL.git" "8511e7c15fbe1fbe5eb25e9f89932ac1ecd546a0"
    Confirm-PinnedRepository "RLPortfolio" "https://github.com/CaioSBC/RLPortfolio.git" "071cf75da9ddf95ec62210627b971086eec6f18f"

    if (-not $CheckOnly) {
        Invoke-Checked $EnvironmentPython @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Checked $EnvironmentPython @("-m", "pip", "install", "-e", ".[data,dev]")
    }
    Invoke-Checked $EnvironmentPython @("-c", "from rlasset.finrl_adapter import load_finrl_agent; _, p = load_finrl_agent(); print('FinRL import/hash verification:', p['commit'])")
    Write-Host "Ready. Python: $EnvironmentPython"
    Write-Host "Run tests: & '$EnvironmentPython' -m pytest -q"
} finally {
    Pop-Location
}
