param([string]$Python = 'python', [string]$Npm = 'npm')
$ErrorActionPreference = 'Stop'
$mvpProject = Split-Path -Parent $PSScriptRoot
$mvpEnv = Join-Path $mvpProject '.venv'
$mvpPython = Join-Path $mvpEnv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $mvpPython)) {
    & $Python -B -c 'import sys; assert sys.version_info[:2] == (3,11), "需要 Python 3.11"'
    if ($LASTEXITCODE -ne 0) { throw 'Python 版本不匹配' }
    & $Python -B -m venv $mvpEnv
    if ($LASTEXITCODE -ne 0) { throw '虚拟环境创建失败' }
    & $mvpPython -B -m pip --isolated install --no-cache-dir --no-compile --index-url https://pypi.org/simple -r (Join-Path $mvpProject 'requirements-mvp-lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Python 依赖安装失败' }
}
Push-Location (Join-Path $mvpProject 'web')
try {
    & $Npm ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw '前端依赖安装失败' }
    & $Npm run build
    if ($LASTEXITCODE -ne 0) { throw '前端构建失败' }
} finally { Pop-Location }
$mvpEntry = Join-Path $mvpProject 'scripts\run_mvp.py'
Write-Output '环境已准备完成。在 PowerShell 中运行以下命令，然后打开 http://127.0.0.1:8765：'
Write-Output ("& '" + $mvpPython.Replace("'", "''") + "' -X utf8 -B '" + $mvpEntry.Replace("'", "''") + "'")
