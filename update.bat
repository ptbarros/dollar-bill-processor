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
if exist "patterns_v2.yaml" (
    echo Backing up your patterns_v2.yaml...
    copy /y "patterns_v2.yaml" "patterns_v2.yaml.backup" >nul
)
if exist "config.yaml" (
    echo Backing up your config.yaml...
    copy /y "config.yaml" "config.yaml.backup" >nul
)
if exist "patterns.txt" (
    echo Backing up your patterns.txt...
    copy /y "patterns.txt" "patterns.txt.backup" >nul
)

:: Copy updated files (the zip extracts to dollar-bill-processor-main/)
echo Installing updates...
set "SOURCE_DIR=%TEMP_DIR%\dollar-bill-processor-main"

:: Copy all files except directories we want to preserve
:: Note: YAML files are handled separately to preserve user_patterns.yaml
for %%F in ("%SOURCE_DIR%\*.py" "%SOURCE_DIR%\*.bat" "%SOURCE_DIR%\*.sh" "%SOURCE_DIR%\*.txt" "%SOURCE_DIR%\*.md" "%SOURCE_DIR%\*.pt") do (
    if exist "%%F" (
        copy /y "%%F" "." >nul 2>&1
    )
)

:: Copy specific YAML files (not user_patterns.yaml or user_settings.yaml)
for %%F in (patterns_v2.yaml config.yaml) do (
    if exist "%SOURCE_DIR%\%%F" (
        copy /y "%SOURCE_DIR%\%%F" "." >nul 2>&1
    )
)

:: Copy requirements.txt specifically
if exist "%SOURCE_DIR%\requirements.txt" (
    copy /y "%SOURCE_DIR%\requirements.txt" "." >nul
)

:: Copy gui folder (for GUI support)
if exist "%SOURCE_DIR%\gui" (
    echo Updating GUI files...
    if not exist "gui" mkdir "gui"
    xcopy /y /e /q "%SOURCE_DIR%\gui\*" "gui\" >nul 2>&1
)

:: Copy patterns folder (core Lua patterns, Nicks library, and helpers)
if exist "%SOURCE_DIR%\patterns" (
    echo Updating pattern files...
    if not exist "patterns" mkdir "patterns"
    if not exist "patterns\core" mkdir "patterns\core"
    if not exist "patterns\Nicks" mkdir "patterns\Nicks"
    if not exist "patterns\lib" mkdir "patterns\lib"
    if not exist "patterns\user" mkdir "patterns\user"
    :: Copy core patterns, Nicks library, and lib (but not user patterns)
    xcopy /y /e /q "%SOURCE_DIR%\patterns\core\*" "patterns\core\" >nul 2>&1
    xcopy /y /e /q "%SOURCE_DIR%\patterns\Nicks\*" "patterns\Nicks\" >nul 2>&1
    xcopy /y /e /q "%SOURCE_DIR%\patterns\lib\*" "patterns\lib\" >nul 2>&1
)

:: Clean up temp files
echo Cleaning up...
del /f "%TEMP_ZIP%" 2>nul
rmdir /s /q "%TEMP_DIR%" 2>nul

:: Install any new dependencies
if exist "venv\Scripts\activate.bat" (
    echo.
    echo Checking for new dependencies...
    call venv\Scripts\activate.bat
    pip install -q -r requirements.txt
)

echo.
echo ============================================
echo   UPDATE COMPLETE!
echo ============================================
echo.
echo Your previous config files were backed up to:
echo   patterns_v2.yaml.backup  (pattern definitions)
echo   config.yaml.backup       (crop settings)
echo.
echo If you had custom patterns in patterns_v2.yaml, you may want
echo to merge them from the backup file.
echo.
pause
