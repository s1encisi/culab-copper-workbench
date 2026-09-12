param(
    [string]$Python = '../.analysis_work/venvs/copper-mvp/Scripts/python.exe'
)
$ErrorActionPreference = 'Stop'
$OptProject = Split-Path -Parent $PSScriptRoot
$OptTarget = Join-Path $OptProject 'runs/dependencies/numba-0.61.2'
$OptRequirements = Join-Path $OptProject 'requirements-optimizer-methods.txt'
& $Python -X utf8 -B -m pip install --disable-pip-version-check --no-deps --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --target $OptTarget -r $OptRequirements
if ($LASTEXITCODE -ne 0) { throw 'Optimizer dependency installation failed.' }
