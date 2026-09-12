param(
    [string]$Python = '../.analysis_work/venvs/copper-mvp/Scripts/python.exe',
    [string]$RuntimeDirectory = ''
)
$ErrorActionPreference = 'Stop'
$SymbolicProject = Split-Path -Parent $PSScriptRoot
if (-not $RuntimeDirectory) { $RuntimeDirectory = Join-Path $SymbolicProject 'runs/dependencies/pysr-1.5.9' }
$env:COPPER_SYMBOLIC_RUNTIME_DIR = $RuntimeDirectory
& $Python -X utf8 -B -m pip install --disable-pip-version-check --no-deps --only-binary=:all: --require-hashes --index-url https://pypi.org/simple --target $RuntimeDirectory -r (Join-Path $SymbolicProject 'requirements-symbolic-models.txt')
if ($LASTEXITCODE -ne 0) { throw 'Symbolic Python dependency installation failed.' }
& $Python -X utf8 -B (Join-Path $PSScriptRoot 'install_symbolic_julia.py')
if ($LASTEXITCODE -ne 0) { throw 'Julia runtime installation failed.' }
& $Python -X utf8 -B (Join-Path $PSScriptRoot 'initialize_symbolic_runtime.py')
if ($LASTEXITCODE -ne 0) { throw 'Julia package initialization failed.' }
