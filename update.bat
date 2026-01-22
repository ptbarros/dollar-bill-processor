@echo off
setlocal EnableDelayedExpansion

:: =============================================================================
:: Dollar Bill Processor - Update Script (No Git Required)
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

set "REPO_URL=https://github.com/ptbarros/dollar-bill-processor/archive/refs/heads/main.zip"
set "TEMP_ZIP=%TEMP%\dollar-bill-update.zip"
set "TEMP_DIR=%TEMP%\dollar-bill-update"

echo Checking for updates...
echo.

:: Clean up any previous temp files
if exist "%TEMP_ZIP%" del /f "%TEMP_ZIP%"
if exist "%TEMP_DIR%" rmdir /s /q "%TEMP_DIR%"

:: Download the latest version using PowerShell
echo Downloading latest version...
powershell -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%REPO_URL%' -OutFile '%TEMP_ZIP%' -UseBasicParsing } catch { exit 1 }"

if errorlevel 1 (
    echo.
    echo ERROR: Could not download updates.
    echo Check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

:: Create temp directory
mkdir "%TEMP_DIR%" 2>nul

:: Extract the zip using PowerShell
echo Extracting files...
powershell -Command "Expand-Archive -Path '%TEMP_ZIP%' -DestinationPath '%TEMP_DIR%' -Force"

if errorlevel 1 (
    echo.
    echo ERROR: Could not extract update files.
    echo.
    pause
    exit /b 1
)

:: Back up user config files if they exist
if exist "patterns.txt" (
    echo Backing up your patterns.txt...
    copy /y "patterns.txt" "patterns.txt.backup" >nul
)
if exist "config.yaml" (
    echo Backing up your config.yaml...
    copy /y "config.yaml" "config.yaml.backup" >nul
)

:: Copy updated files (the zip extracts to dollar-bill-processor-main/)
echo Installing updates...
set "SOURCE_DIR=%TEMP_DIR%\dollar-bill-processor-main"

:: Copy all files except directories we want to preserve
for %%F in ("%SOURCE_DIR%\*.py" "%SOURCE_DIR%\*.bat" "%SOURCE_DIR%\*.sh" "%SOURCE_DIR%\*.txt" "%SOURCE_DIR%\*.yaml" "%SOURCE_DIR%\*.md") do (
    if exist "%%F" (
        copy /y "%%F" "." >nul 2>&1
    )
)

:: Copy requirements.txt specifically
if exist "%SOURCE_DIR%\requirements.txt" (
    copy /y "%SOURCE_DIR%\requirements.txt" "." >nul
)

:: Clean up temp files
echo Cleaning up...
del /f "%TEMP_ZIP%" 2>nul
rmdir /s /q "%TEMP_DIR%" 2>nul

echo.
echo ============================================
echo   UPDATE COMPLETE!
echo ============================================
echo.
echo Your previous patterns.txt and config.yaml were backed up to:
echo   patterns.txt.backup
echo   config.yaml.backup
echo.
echo If you had custom patterns, you may want to merge them
echo from the backup files.
echo.
pause
