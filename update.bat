@echo off
setlocal

:: =============================================================================
:: Dollar Bill Processor - Update Script
:: =============================================================================
::
:: Double-click this file to download the latest updates.
::
:: =============================================================================

title Dollar Bill Processor - Update

:: Change to the directory where this batch file is located
cd /d "%~dp0"

echo.
echo ============================================
echo   DOLLAR BILL PROCESSOR - UPDATE
echo ============================================
echo.

:: Check if git is installed
where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed!
    echo.
    echo Please install Git from:
    echo   https://git-scm.com/download/win
    echo.
    echo During installation, use the default options.
    echo.
    pause
    exit /b 1
)

echo Checking for updates...
echo.

:: Fetch and pull latest changes
git fetch origin main
if errorlevel 1 (
    echo.
    echo ERROR: Could not connect to GitHub.
    echo Check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

:: Check if there are updates
git status -uno | findstr /C:"behind" >nul
if errorlevel 1 (
    echo Already up to date! No changes needed.
    echo.
    pause
    exit /b 0
)

:: Pull the updates
echo Updates found! Downloading...
echo.
git pull origin main

if errorlevel 1 (
    echo.
    echo ERROR: Could not download updates.
    echo.
    echo If you have made local changes, they may conflict.
    echo Contact support for help.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   UPDATE COMPLETE!
echo ============================================
echo.
echo You can now close this window.
echo.
pause
