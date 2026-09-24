param(
    [string]$Python = (Join-Path $PSScriptRoot '../.venv/Scripts/python.exe')
)
$ErrorActionPreference = 'Stop'
$PlatProject = Split-Path -Parent $PSScriptRoot
$PlatTarget = Join-Path $PlatProject 'runs/dependencies/platypus-1.4.1'
$PlatRequirements = Join-Path $PlatProject 'requirements-platypus-methods.txt'
& $Python -X utf8 -B -m pip install --no-cache-dir --no-compile --disable-pip-version-check --no-deps --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --target $PlatTarget -r $PlatRequirements
if ($LASTEXITCODE -ne 0) { throw 'Platypus dependency installation failed.' }
