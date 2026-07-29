@echo off
chcp 65001 >nul
title CS2 AutoAim

REM ==========================================
REM 便携版 - 自动适配任意路径
REM ==========================================
set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"

REM 设置 Python 根目录 (stdlib + DLLs 都在 venv 里)
set "PYTHONHOME=%APP_DIR%\venv"

REM CUDA DLLs (PyTorch 自带)
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
echo ║  按住 [鼠标侧后键] = 自瞄, 松开停止      ║
echo ║  Q=退出  F3=切换持续自瞄                 ║
echo ╚══════════════════════════════════════════╝
echo.
"%PYTHON%" src\inference_onnx.py %*