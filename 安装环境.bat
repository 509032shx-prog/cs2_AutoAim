@echo off
chcp 65001 >nul
title CS2 AutoAim - 在线安装

echo.
echo   ╔══════════════════════════════════════════╗
echo   ║   CS2 AutoAim — 在线安装 (需联网)        ║
echo   ╚══════════════════════════════════════════╝
echo.

set "ROOT=%~dp0"
set "VENV=%ROOT%venv"

REM ==========================================
REM Step 1: Check Python
REM ==========================================
echo [1/4] 检查 Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [X] 未找到 Python！请先安装 Python 3.10~3.12
    echo      下载: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   [√] Python %PYVER%
echo.

REM ==========================================
REM Step 2: Create venv
REM ==========================================
echo [2/4] 创建虚拟环境...
if not exist "%VENV%" (
    python -m venv --copies "%VENV%"
    echo   [√] 虚拟环境已创建 (copies 模式)
) else (
    echo   [√] 虚拟环境已存在, 跳过
)
call "%VENV%\Scripts\activate.bat"
python -m pip install --upgrade pip -q 2>&1 >nul
echo.

REM ==========================================
REM Step 3: Install PyTorch CUDA 12.8
REM ==========================================
echo [3/4] 安装 PyTorch CUDA 12.8 (约 2.8GB, 需联网)...
echo   下载中, 请耐心等待...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    echo   [X] PyTorch 安装失败! 请检查网络
    pause
    exit /b 1
)
echo   [√] PyTorch + CUDA 12.8 安装完成
echo.

REM ==========================================
REM Step 4: Install other dependencies
REM ==========================================
echo [4/4] 安装其余依赖...
cd /d "%ROOT%"
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo   [X] 依赖安装失败!
    pause
    exit /b 1
)
echo   [√] 依赖安装完成
echo.

REM ==========================================
REM Verify
REM ==========================================
echo ╔══════════════════════════════════════════╗
echo ║  验证安装...                              ║
echo ╚══════════════════════════════════════════╝
python -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')" 2>nul
python -c "import onnxruntime; print(f'  ONNX Runtime {onnxruntime.__version__}  Providers={onnxruntime.get_available_providers()}')" 2>nul
python -c "import cv2; print(f'  OpenCV {cv2.__version__}')" 2>nul
python -c "import ultralytics; from ultralytics import YOLO; m=YOLO(r'%ROOT%models\best.pt'); print(f'  Ultralytics OK  模型={m.names}')" 2>nul

if exist "%ROOT%models\cs2_best.onnx" (
    echo   [√] ONNX 模型已就绪
) else (
    echo   [!] ONNX 模型不存在, 正在导出...
    python scripts\export_onnx.py
)

echo.
echo ╔══════════════════════════════════════════╗
echo ║  [√] 安装完成！                           ║
echo ╠══════════════════════════════════════════╣
echo ║  双击 "开始运行.bat" 启动推理              ║
echo ║  双击 "开始运行+自瞄.bat" 启动+自动瞄准     ║
echo ╚══════════════════════════════════════════╝
echo.
pause
