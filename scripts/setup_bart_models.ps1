param([string]$Python = (Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'))
$ErrorActionPreference = 'Stop'
$BartProject = Split-Path -Parent $PSScriptRoot
$BartTarget = Join-Path $BartProject 'runs/dependencies/pymc-bart-0.7.0'
$BartRequirements = Join-Path $BartProject 'requirements-bart-models.txt'
& $Python -X utf8 -B -m pip install --no-cache-dir --no-compile --disable-pip-version-check --no-deps --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --target $BartTarget -r $BartRequirements
if ($LASTEXITCODE -ne 0) { throw 'BART dependency installation failed.' }
