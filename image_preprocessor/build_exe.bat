@echo off
REM ============================================================
REM  Image Preprocessor - Build Windows EXE
REM  Run this script on a Windows machine with Python installed.
REM ============================================================
echo.
echo  Image Preprocessor - EXE Builder
echo  ===================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Install / upgrade build dependencies
echo [1/3] Installing dependencies...
pip install opencv-python numpy pyinstaller --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

REM Build EXE
echo [2/3] Building EXE (this may take 1-3 minutes)...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name "ImagePreprocessor" ^
    --add-data "src;src" ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    gui_app.py

if errorlevel 1 (
    echo [ERROR] Build failed. Check the output above.
    pause
    exit /b 1
)

REM Copy icon if exists to dist/
if exist icon.ico copy icon.ico dist\icon.ico >nul 2>&1

echo [3/3] Cleaning up build files...
rmdir /s /q build >nul 2>&1
del ImagePreprocessor.spec >nul 2>&1

echo.
echo  ====================================================
echo   Build COMPLETE!
echo   EXE location: dist\ImagePreprocessor.exe
echo.
echo   To distribute, copy the entire dist\ folder.
echo   Users can run ImagePreprocessor.exe directly —
echo   no Python or installation required.
echo  ====================================================
echo.
pause
