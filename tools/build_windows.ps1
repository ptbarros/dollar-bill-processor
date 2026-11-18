<#
.SYNOPSIS
  Build one Windows edition of Dollar Detective locally (for TESTING only).

.DESCRIPTION
  Mirrors the Windows jobs in .github/workflows/build.yml step for step, so
  "it built fine locally" means the same thing as a CI build. KEEP THE TWO IN
  SYNC: if you change a build step in build.yml, change it here too.

  Each edition gets its own venv (.venv-build-<edition>) because the three
  onnxruntime variants (openvino / directml / torch+CUDA) conflict. The venv is
  created on the first run and reused afterwards.

  Local builds are for testing only. Never upload them to GitHub Releases or
  tag anything from them: official releases come from the CI tag build, which
  is what the in-app updater downloads.

.PARAMETER Edition
  openvino (standard), directml or cuda.

.PARAMETER SkipInstaller
  Stop after the PyInstaller build and the frozen self-test (no Inno Setup).

.PARAMETER SkipDeps
  Reuse the existing venv as is (skip the pip steps). Faster for rebuilds when
  requirements haven't changed. Ignored if the venv doesn't exist yet.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\build_windows.ps1 -Edition directml
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('openvino', 'directml', 'cuda')]
    [string]$Edition,
    [switch]$SkipInstaller,
    [switch]$SkipDeps
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$timings = [System.Collections.Generic.List[object]]::new()
$runStart = Get-Date

function Step([string]$Name, [scriptblock]$Body) {
    Write-Host ""
    Write-Host "=== $Name" -ForegroundColor Cyan
    $t = Get-Date
    & $Body
    $timings.Add([pscustomobject]@{ Step = $Name; Seconds = [math]::Round(((Get-Date) - $t).TotalSeconds, 1) })
}

# Run a native command and fail on a non-zero exit code. Windows PowerShell 5.1
# turns native stderr into errors when $ErrorActionPreference is Stop, and pip
# writes harmless warnings there, so relax it just for the call.
function Exec([scriptblock]$Cmd, [switch]$AllowFail) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Cmd; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $old }
    if ($code -ne 0 -and -not $AllowFail) { throw "Command failed (exit $code): $Cmd" }
}

# ---------------------------------------------------------------- prerequisites
function Find-Python312 {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $exe = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
    }
    foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe", "$env:ProgramFiles\Python312\python.exe")) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Find-Iscc {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($k in @('HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1',
                     'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1',
                     'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1')) {
        $loc = (Get-ItemProperty $k -ErrorAction SilentlyContinue).InstallLocation
        if ($loc) { $candidates += (Join-Path $loc 'ISCC.exe') }
    }
    foreach ($c in $candidates) { if ($c -and (Test-Path $c)) { return $c } }
    return $null
}

$Csc = "$env:WINDIR\Microsoft.NET\Framework\v4.0.30319\csc.exe"
$Python = Find-Python312
$Iscc = if ($SkipInstaller) { $null } else { Find-Iscc }
$missing = @()
if (-not $Python) { $missing += 'Python 3.12   ->  winget install -e --id Python.Python.3.12' }
if (-not (Test-Path $Csc)) { $missing += ".NET Framework 4 compiler ($Csc) -> part of Windows; enable .NET Framework 4.8 in Windows Features" }
if (-not $SkipInstaller -and -not $Iscc) { $missing += 'Inno Setup 6  ->  winget install -e --id JRSoftware.InnoSetup   (or use -SkipInstaller)' }
if ($missing) {
    Write-Host "Missing prerequisites:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host "(Git is needed to get/update the repo:  winget install -e --id Git.Git)"
    exit 1
}

$Venv = Join-Path $Repo ".venv-build-$Edition"
$VPy = Join-Path $Venv 'Scripts\python.exe'
$Spec = if ($Edition -eq 'cuda') { 'DollarDetective-cuda.spec' } else { 'DollarDetective.spec' }
Write-Host "Dollar Detective local build: edition=$Edition  python=$Python  venv=$Venv"

# Run pip/pyinstaller/python from the edition's venv (CI's "pip"/"python" are
# the job's setup-python interpreter; here the venv plays that role).
$env:VIRTUAL_ENV = $Venv
$env:Path = "$Venv\Scripts;$env:Path"

# ------------------------------------------------------------------ venv + deps
$newVenv = -not (Test-Path $VPy)

# LOCAL-ONLY step (not in build.yml). In a reused venv, `pip install -r
# requirements.txt` puts plain onnxruntime back, and uninstalling it deletes
# files it shares with the onnxruntime-openvino/-directml wheel (same
# `onnxruntime` folder). pip then says the variant is "already satisfied" and
# leaves it broken. CI always starts from a fresh runner, so it never hits this.
function Repair-OrtVariant([string]$Requirement) {
    if (-not $newVenv) { Exec { pip install --force-reinstall --no-deps $Requirement } }
}
if ($newVenv) {
    Step "Create venv $Venv" { Exec { & $Python -m venv $Venv } }
}

if ($SkipDeps -and -not $newVenv) {
    Write-Host "`n=== Skipping dependency install (-SkipDeps)" -ForegroundColor Yellow
} elseif ($Edition -eq 'openvino') {
    Step 'Install dependencies (default = OpenVINO)' {
        Exec { python -m pip install --upgrade pip }
        Exec { pip install -r requirements.txt }
        Exec { pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml } -AllowFail
        Exec { pip install "onnxruntime-openvino>=1.17.0" }
        Repair-OrtVariant 'onnxruntime-openvino>=1.17.0'
        Exec { python -c "import onnxruntime as o; p=o.get_available_providers(); print('providers:',p); assert 'OpenVINOExecutionProvider' in p, 'OpenVINO EP missing'" }
        Exec { pip uninstall -y QScintilla PyQt5 PyQt5-Qt5 PyQt5-sip } -AllowFail
        Exec { pip install pyinstaller }
    }
} elseif ($Edition -eq 'directml') {
    Step 'Install dependencies (DirectML fallback edition)' {
        Exec { python -m pip install --upgrade pip }
        Exec { pip install -r requirements.txt }
        Exec { pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-openvino } -AllowFail
        Exec { pip install "onnxruntime-directml>=1.17.0" }
        Repair-OrtVariant 'onnxruntime-directml>=1.17.0'
        Exec { python -c "import onnxruntime as o; p=o.get_available_providers(); print('providers:',p); assert 'DmlExecutionProvider' in p, 'DirectML EP missing'" }
        Exec { pip uninstall -y QScintilla PyQt5 PyQt5-Qt5 PyQt5-sip } -AllowFail
        Exec { pip install pyinstaller }
    }
} else {
    Step 'Install dependencies (CUDA: torch + EasyOCR)' {
        Exec { python -m pip install --upgrade pip }
        Exec { pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 }
        Exec { pip install -r requirements-cuda.txt }
        Exec { pip uninstall -y opencv-python } -AllowFail
        Exec { pip install --force-reinstall --no-deps opencv-python-headless }
        Exec { pip uninstall -y QScintilla PyQt5 PyQt5-Qt5 PyQt5-sip } -AllowFail
        Exec { pip install pyinstaller }
        Exec { python -c "import torch; print('torch', torch.__version__, 'cuda build:', torch.version.cuda)" }
        # CI also runs `pip cache purge` here to save runner disk; skipped locally
        # so rebuilds reuse the cached wheels.
    }
}

if ($Edition -eq 'cuda') {
    Step 'Fetch EasyOCR models (bundle them so first run never downloads)' {
        Exec { python -c "import easyocr; easyocr.Reader(['en'], gpu=False, model_storage_directory='easyocr_models', download_enabled=True, verbose=False)" }
        Get-ChildItem easyocr_models | Format-Table Name, Length -AutoSize | Out-Host
    }
}

# ---------------------------------------------------------------- TWAIN helper
Step 'Build TWAIN helper (32-bit, built-in .NET Framework compiler)' {
    New-Item -ItemType Directory -Force scanner_helpers | Out-Null
    Exec { & $Csc /nologo /platform:x86 /target:exe /optimize+ `
        /r:System.Windows.Forms.dll /out:scanner_helpers\dd_twain_helper.exe tools\twain_helper\TwainHelper.cs }
    Get-Item scanner_helpers\dd_twain_helper.exe | Format-Table Name, Length, LastWriteTime -AutoSize | Out-Host
}

# ------------------------------------------------------------------ PyInstaller
Step "Build onedir (PyInstaller, $Spec)" {
    # A fresh CI checkout has no build/ or dist/DollarDetective. Clear them so a
    # previous edition's files can't leak into this installer. Other editions'
    # finished *-setup.exe files in dist/ are kept.
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    if (Test-Path dist\DollarDetective) { Remove-Item -Recurse -Force dist\DollarDetective }
    Exec { pyinstaller --noconfirm --clean $Spec }
}

# CI runs this for directml only; locally it runs for every edition.
Step 'Verify frozen build loads patterns + reads a serial (self-test)' {
    $old = @{ DBP_SELFTEST = $env:DBP_SELFTEST; QT_QPA_PLATFORM = $env:QT_QPA_PLATFORM }
    $env:DBP_SELFTEST = Join-Path $Repo 'assets\samples\five_dollar\five_dollar.jpg'
    $env:QT_QPA_PLATFORM = 'offscreen'
    try {
        $p = Start-Process -FilePath "dist\DollarDetective\DollarDetective.exe" -Wait -PassThru -NoNewWindow
    } finally {
        $env:DBP_SELFTEST = $old.DBP_SELFTEST; $env:QT_QPA_PLATFORM = $old.QT_QPA_PLATFORM
    }
    if ($p.ExitCode -ne 0) { throw "frozen self-test FAILED (exit $($p.ExitCode))" }
    Write-Host "frozen self-test PASSED" -ForegroundColor Green
}

# -------------------------------------------------------------------- installer
$setup = $null
if (-not $SkipInstaller) {
    Step 'Build installer (Inno Setup)' {
        $ver = (python -c "from dollar_detective.version import __version__; print(__version__)").Trim()
        if ($LASTEXITCODE -ne 0 -or -not $ver) { throw "Could not read dollar_detective/version.py" }
        $isccArgs = @("/DMyAppVersion=$ver")
        if ($Edition -eq 'cuda') { $isccArgs += '/DEditionTag=-cuda', '/DEditionName= (CUDA)' }
        if ($Edition -eq 'directml') { $isccArgs += '/DEditionTag=-directml', '/DEditionName= (DirectML)' }
        $isccArgs += 'installer\DollarDetective.iss'
        Exec { & $Iscc @isccArgs }
        $tag = if ($Edition -eq 'openvino') { '' } else { "-$Edition" }
        $script:setup = Join-Path $Repo "dist\DollarDetective-$ver$tag-setup.exe"
        if (-not (Test-Path $script:setup)) { throw "Installer not found: $script:setup" }
    }
}

# ---------------------------------------------------------------------- summary
Write-Host ""
Write-Host "=== Build finished: $Edition" -ForegroundColor Green
$timings | Format-Table -AutoSize | Out-Host
Write-Host ("Total: {0:mm\:ss}" -f ((Get-Date) - $runStart))
if ($setup) {
    Write-Host "Installer (local TEST build -- do not publish): $setup" -ForegroundColor Green
} else {
    Write-Host "App folder: $(Join-Path $Repo 'dist\DollarDetective')"
}
