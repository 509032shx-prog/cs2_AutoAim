@echo off
chcp 65001 >nul
title CS2 AutoAim + Auto

REM ==========================================
REM 便携版 - 自动适配任意路径
REM ==========================================
set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"

set "PYTHONHOME=%APP_DIR%\venv"

if exist "%PYTHONHOME%\Lib\site-packages\torch\lib\cublas64_12.dll" (
    set "PATH=%PYTHONHOME%\Lib\site-packages\torch\lib;%PATH%"
)

set "PYTHON=%PYTHONHOME%\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [X] Python 环境损坏，请重新解压
    pause
    exit /b 1
)

if not exist "%APP_DIR%\models\cs2_best.onnx" (
    echo [X] 模型文件缺失 (models\cs2_best.onnx)
    pause
    exit /b 1
)

cd /d "%APP_DIR%"
echo.
echo ╔══════════════════════════════════════════╗
echo ║  持续自瞄模式 (F3切换回鼠标侧键)        ║
echo ║  按住 [鼠标侧后键] 也可瞄               ║
echo ╚══════════════════════════════════════════╝
echo.
"%PYTHON%" src/inference_onnx.py --aimbot %*