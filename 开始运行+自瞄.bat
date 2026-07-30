@echo off
chcp 65001 >nul
title CS2 AutoAim + Auto
set PYTHONIOENCODING=utf-8

set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"
set "PYTHONHOME=%APP_DIR%\venv"

if exist "%PYTHONHOME%\Lib\site-packages\torch\lib\cublas64_12.dll" (
    set "PATH=%PYTHONHOME%\Lib\site-packages\torch\lib;%PATH%"
)

set "PYTHON=%PYTHONHOME%\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [X] Python not found, please re-extract
    pause
    exit /b 1
)

if not exist "%APP_DIR%\models\cs2_best.onnx" (
    echo [X] Model not found: models\cs2_best.onnx
    pause
    exit /b 1
)

cd /d "%APP_DIR%"
echo.
echo ============================================
echo   CS2 AutoAim - Auto-aim mode - F3 to toggle
echo ============================================
echo.
"%PYTHON%" src/inference_onnx.py --aimbot %*