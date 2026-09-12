param(
    [string]$Python = '../.analysis_work/venvs/copper-mvp/Scripts/python.exe'
)
$ErrorActionPreference = 'Stop'
$StatProject = Split-Path -Parent $PSScriptRoot
$StatTarget = Join-Path $StatProject 'runs/dependencies/statsmodels-0.14.6'
$StatRequirements = Join-Path $StatProject 'requirements-statistical-models.txt'
& $Python -X utf8 -B -m pip install --disable-pip-version-check --no-deps --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --target $StatTarget -r $StatRequirements
if ($LASTEXITCODE -ne 0) { throw 'Statistics dependency installation failed.' }
