@echo off
setlocal

:: =============================================================================
:: Dollar Bill Processor - Windows Setup Script
:: =============================================================================
::
:: Run this ONCE to set up the environment.
:: After setup, use run_processor.bat to process bills.
::
:: =============================================================================

title Dollar Bill Processor - Setup

:: Change to the directory where this batch file is located
cd /d "%~dp0"

echo.
echo ============================================
echo   DOLLAR BILL PROCESSOR - SETUP
echo ============================================
echo.

:: Check if 64-bit Python is installed (required for PyTorch)
py -3-64 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: 64-bit Python is not installed!
    echo.
    echo This tool requires 64-bit Python because PyTorch does not
    echo support 32-bit Windows.
    echo.
    echo Please install 64-bit Python 3.10 or higher from:
    echo   https://www.python.org/downloads/
    echo.
    echo Download the "Windows installer (64-bit)" version.
    echo IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    echo Note: You can have both 32-bit and 64-bit Python installed.
    echo The py launcher will select the correct version automatically.
    echo.
    pause
    exit /b 1
)

echo 64-bit Python found:
py -3-64 --version
echo.

:: Check if venv already exists
if exist "venv" (
    echo Virtual environment already exists.
    echo.
    set /p RECREATE="Recreate it? (y/N): "
    if /i not "%RECREATE%"=="y" goto :install_deps
    echo Removing old environment...
    rmdir /s /q venv
)

:: Create virtual environment using 64-bit Python
echo Creating virtual environment...
py -3-64 -m venv venv
if errorlevel 1 (
    echo.
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)
echo Done.
echo.

:install_deps
:: Activate and install dependencies
echo Activating environment...
call venv\Scripts\activate.bat

:: Check for NVIDIA GPU and install CUDA-enabled PyTorch if available
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if errorlevel 1 goto :no_gpu

echo   NVIDIA GPU detected!
echo   Installing CUDA-enabled PyTorch (cu128 for RTX 50-series support)...
echo.
pip uninstall torch torchvision -y >nul 2>&1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo.
    echo WARNING: Failed to install CUDA PyTorch, falling back to CPU version.
    echo.
)
goto :install_deps_main

:no_gpu
echo   No NVIDIA GPU detected - will use CPU-only PyTorch.
echo.

:install_deps_main
echo Installing remaining dependencies...
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

:: Verify GPU support and check compute capability compatibility
echo.
echo Checking GPU support...
python -c "import torch; gpu=torch.cuda.is_available(); print('  PyTorch CUDA:', gpu); exec('if gpu:\n try:\n  print(\"  Device:\", torch.cuda.get_device_name(0))\n  cap = torch.cuda.get_device_capability(0)\n  print(f\"  Compute capability: sm_{cap[0]}{cap[1]}\")\n  t = torch.randn(1, device=\"cuda\"); _ = t + t\n  print(\"  CUDA kernel test: PASSED\")\n except Exception as e:\n  print(f\"  WARNING: CUDA kernel test FAILED: {e}\")\n  print(\"  GPU will not be used. Processing will fall back to CPU.\")\n  print(\"  If using an RTX 50-series GPU, try reinstalling with:\")\n  print(\"    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\")\nelse:\n print(\"  CPU-only mode\")')"
echo.

echo ============================================
echo   SETUP COMPLETE!
echo ============================================
echo.
echo You can now use:
echo   - run_processor.bat  : Command-line processing
echo   - run_gui.bat        : Graphical interface (recommended)
echo.
pause
