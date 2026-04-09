@echo off
REM ============================================================
REM  Image Preprocessor - Quick Run (requires Python installed)
REM  Double-click this file to start the GUI.
REM ============================================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Please install Python 3.9+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check and install dependencies if needed
python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo [INFO] First run — installing required packages...
    pip install opencv-python numpy
    if errorlevel 1 (
        echo [ERROR] Failed to install packages.
        pause
        exit /b 1
    )
    echo [INFO] Packages installed successfully.
)

REM Run the GUI
python gui_app.py
