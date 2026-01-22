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

echo.
echo Installing dependencies (this may take a few minutes)...
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   SETUP COMPLETE!
echo ============================================
echo.
echo You can now use run_processor.bat to process bills.
echo.
echo To edit patterns, open patterns_v2.yaml in Notepad.
echo.
pause
