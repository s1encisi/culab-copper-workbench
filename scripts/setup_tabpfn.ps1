param(
    [string]$Python = (Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'),
    [string]$RuntimeDirectory = ''
)
$ErrorActionPreference='Stop'
$TabPFNProject=Split-Path -Parent $PSScriptRoot
if (-not $RuntimeDirectory) { $RuntimeDirectory=Join-Path $TabPFNProject 'runs/dependencies/tabpfn-2.1.3' }
$env:COPPER_TABPFN_RUNTIME_DIR=$RuntimeDirectory
& $Python -X utf8 -B -m pip install --no-cache-dir --no-compile --disable-pip-version-check --no-deps --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --target $RuntimeDirectory -r (Join-Path $TabPFNProject 'requirements-tabpfn-isolated.txt')
if ($LASTEXITCODE -ne 0) { throw 'TabPFN isolated dependencies failed.' }
& $Python -X utf8 -B (Join-Path $PSScriptRoot 'install_tabpfn_weights.py')
if ($LASTEXITCODE -ne 0) { throw 'TabPFN weight verification failed.' }
