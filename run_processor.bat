@echo off
setlocal

:: =============================================================================
:: Dollar Bill Processor - Easy Run Script
:: =============================================================================
::
:: USAGE:
::   Option 1: Double-click this file and enter folder paths when prompted
::   Option 2: Drag a folder onto this file to process it
::
:: =============================================================================

title Dollar Bill Processor

:: Change to the directory where this batch file is located
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo ERROR: Virtual environment not found!
    echo.
    echo Please run setup_windows.bat first to install dependencies.
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

echo.
echo ============================================
echo   DOLLAR BILL PROCESSOR
echo ============================================
echo.

:: Check if a folder was dragged onto the batch file
if not "%~1"=="" (
    set "INPUT_DIR=%~1"
    goto :set_output
)
goto :prompt_input

:set_output
:: Remove trailing backslash if present
if "%INPUT_DIR:~-1%"=="\" set "INPUT_DIR=%INPUT_DIR:~0,-1%"
set "OUTPUT_DIR=%INPUT_DIR%\fancy_bills"
goto :ask_mode

:prompt_input

:: Otherwise, prompt for input
echo Enter the path to your scanned bills folder:
echo (You can drag and drop a folder here)
echo.
set /p INPUT_DIR="Scans folder: "

:: Remove quotes if present
set INPUT_DIR=%INPUT_DIR:"=%

:: Check if folder exists
if not exist "%INPUT_DIR%" (
    echo.
    echo ERROR: Folder not found: %INPUT_DIR%
    echo.
    pause
    exit /b 1
)

:: Set default output directory
set "OUTPUT_DIR=%INPUT_DIR%\fancy_bills"

:ask_mode
echo.
echo Output will be saved to: %OUTPUT_DIR%
echo.
echo ============================================
echo   SELECT MODE
echo ============================================
echo.
echo   1. Fancy only - Only crop bills matching patterns (default)
echo   2. Crop ALL   - Crop every bill (for pre-sorted stacks)
echo.
set /p MODE="Enter choice (1 or 2): "

set "ALL_FLAG="
if "%MODE%"=="2" set "ALL_FLAG=--all"

echo.
echo Press any key to start processing, or Ctrl+C to cancel...
pause >nul

:run
echo.
echo Processing: %INPUT_DIR%
echo Output to:  %OUTPUT_DIR%
if "%MODE%"=="2" echo Mode: Crop ALL bills
echo.

:: Create output directory explicitly (backup in case Python fails)
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo ============================================
echo.

:: Run the processor
python process_production.py "%INPUT_DIR%" --output "%OUTPUT_DIR%" %ALL_FLAG%

echo.
echo ============================================
echo   COMPLETE!
echo ============================================
echo.
echo Results saved to: %OUTPUT_DIR%
echo.
echo Check the summary file for fancy bill positions.
echo.
pause
