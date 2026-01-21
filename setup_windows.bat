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

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.10 or higher from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

echo Python found:
python --version
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

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv
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
echo To edit custom patterns, open patterns.txt in Notepad.
echo.
pause
