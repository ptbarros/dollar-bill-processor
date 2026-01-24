@echo off
setlocal
REM Dollar Bill Processor - GUI Launcher
REM Launches the graphical user interface

title Dollar Bill Processor - GUI

REM Change to the directory where this batch file is located
cd /d "%~dp0"

echo.
echo ============================================
echo   DOLLAR BILL PROCESSOR - GUI
echo ============================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo.
    echo Virtual environment not found!
    echo Please run setup_windows.bat first.
    echo.
    pause
    exit /b 1
)

REM Check if PySide6 is installed (may be missing for older installations)
python -c "import PySide6" 2>nul
if errorlevel 1 (
    echo.
    echo GUI dependencies not found. Installing...
    echo This only needs to happen once.
    echo.
    pip install PySide6 pandas openpyxl
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install GUI dependencies!
        echo Try running: pip install PySide6 pandas openpyxl
        pause
        exit /b 1
    )
    echo.
    echo Dependencies installed successfully!
    echo.
)

REM Launch the GUI
echo Starting GUI...
python run_gui.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause >nul
)
