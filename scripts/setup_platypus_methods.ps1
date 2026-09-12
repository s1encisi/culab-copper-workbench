param(
    [string]$Python = '../.analysis_work/venvs/copper-mvp/Scripts/python.exe'
)
$ErrorActionPreference = 'Stop'
$PlatProject = Split-Path -Parent $PSScriptRoot
$PlatTarget = Join-Path $PlatProject 'runs/dependencies/platypus-1.4.1'
$PlatRequirements = Join-Path $PlatProject 'requirements-platypus-methods.txt'
& $Python -X utf8 -B -m pip install --disable-pip-version-check --no-deps --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --target $PlatTarget -r $PlatRequirements
if ($LASTEXITCODE -ne 0) { throw 'Platypus dependency installation failed.' }
