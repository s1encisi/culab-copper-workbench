param(
    [string]$Python = '../.analysis_work/venvs/copper-mvp/Scripts/python.exe'
)
$ErrorActionPreference = 'Stop'
$NeuralProject = Split-Path -Parent $PSScriptRoot
$NeuralTarget = Join-Path $NeuralProject 'runs/dependencies/torch-botorch-2.8.0'
$NeuralRequirements = Join-Path $NeuralProject 'requirements-neural-bo.txt'
& $Python -X utf8 -B -m pip install --disable-pip-version-check --no-deps --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --target $NeuralTarget -r $NeuralRequirements
if ($LASTEXITCODE -ne 0) { throw 'Neural and Bayesian runtime installation failed.' }
