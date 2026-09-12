param(
    [string]$Python = '../.analysis_work/venvs/copper-mvp/Scripts/python.exe'
)
$ErrorActionPreference = 'Stop'
$SpecialProject = Split-Path -Parent $PSScriptRoot
$SpecialTarget = Join-Path $SpecialProject 'runs/dependencies/specialized-models-v1'
$SpecialRequirements = Join-Path $SpecialProject 'requirements-specialized-models.txt'
& $Python -X utf8 -B -m pip install --disable-pip-version-check --no-deps --no-build-isolation --only-binary=:all: --no-binary=autograd-gamma --require-hashes --index-url https://pypi.org/simple --target $SpecialTarget -r $SpecialRequirements
if ($LASTEXITCODE -ne 0) { throw 'Specialized model dependency installation failed.' }
$CubistTarget = Join-Path $SpecialProject 'runs/dependencies/cubist-1.2.2'
$CubistRequirements = Join-Path $SpecialProject 'requirements-cubist.txt'
& $Python -X utf8 -B -m pip install --disable-pip-version-check --no-deps --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --target $CubistTarget -r $CubistRequirements
if ($LASTEXITCODE -ne 0) { throw 'Cubist dependency installation failed.' }
