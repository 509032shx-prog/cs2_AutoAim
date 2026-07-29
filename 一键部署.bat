@echo off
chcp 65001 >nul
title CS2 AutoAim - 一键部署 (离线版)

echo.
echo   ╔══════════════════════════════════════════╗
echo   ║   CS2 AutoAim — 一键自动部署 (离线)       ║
echo   ╚══════════════════════════════════════════╝
echo.

REM ==========================================
REM Auto-detect offline packages / chunks
REM ==========================================
set "ROOT=%~dp0"
set "VENV=%ROOT%venv"
set "PKGS=%ROOT%offline_packages"
set "CHUNKS=%ROOT%offline_chunks"

REM If only chunks exist, auto-merge first
if not exist "%PKGS%" (
    if exist "%CHUNKS%" (
        echo   ⚠️  检测到分卷包, 正在合并离线包...
        if exist "%CHUNKS%\_合并离线包.bat" (
            pushd "%CHUNKS%"
            call "_合并离线包.bat"
            popd
        )
    )
)

if not exist "%PKGS%" (
    echo   ❌ 离线包不存在! 
    echo      offline_packages\ 和 offline_chunks\ 都没找到
    echo      请从 GitHub Releases 下载离线包
    pause
    exit /b 1
)

REM ==========================================
REM Step 1: Check Python
REM ==========================================
echo [1/6] 检查 Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ❌ 未找到 Python！请先安装 Python 3.10～3.12
    echo      下载: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   ✅ Python %PYVER%
echo.

REM ==========================================
REM Step 2: Check NVIDIA GPU
REM ==========================================
echo [2/6] 检查 NVIDIA GPU...
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >nul 2>&1
if %errorlevel% neq 0 (
    echo   ❌ 未检测到 NVIDIA GPU 或驱动！
    echo      需要: NVIDIA 驱动 ^>= 525
    echo      CPU 推理可用但速度很慢。
    pause
) else (
    for /f "tokens=1,2 delims=," %%a in ('nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2^>nul') do (
        echo   ✅ GPU: %%a
        echo   ✅ 驱动: %%b
    )
)
echo.

REM ==========================================
REM Step 3: Create venv
REM ==========================================
echo [3/6] 创建虚拟环境...
if not exist "%VENV%" (
    python -m venv "%VENV%"
    echo   ✅ 虚拟环境已创建
) else (
    echo   ⚠️  虚拟环境已存在, 跳过
)
call "%VENV%\Scripts\activate.bat"
python -m pip install --upgrade pip -q 2>&1 >nul
echo.

REM ==========================================
REM Step 4: Install PyTorch CUDA 12.8 (从本地)
REM ==========================================
echo [4/6] 安装 PyTorch CUDA 12.8 (从本地, 约 2.8GB)...
REM 先装 torch 再装 torchvision
for %%f in ("%PKGS%\torch-2.11*cu128*.whl") do pip install "%%f" --no-index --no-deps
if %errorlevel% neq 0 (
    echo   ❌ PyTorch 安装失败!
    pause
    exit /b 1
)
for %%f in ("%PKGS%\torchvision-0.26*cu128*.whl") do pip install "%%f" --no-index --no-deps
echo   ✅ PyTorch + CUDA 12.8 安装完成
echo.

REM ==========================================
REM Step 5: Install all other deps (从本地, 跳过已装的 torch)
REM ==========================================
echo [5/6] 安装其余依赖 (从本地)...
REM 先装 torch 的依赖 (torch 已装, --no-deps 跳过)
pip install "%PKGS%\filelock-3.32*" "%PKGS%\typing_extensions-4.16*"^
 "%PKGS%\sympy-1.14*" "%PKGS%\networkx-3.6*" "%PKGS%\jinja2-3.1*"^
 "%PKGS%\fsspec-2026.6*" "%PKGS%\setuptools-83*"^
 --no-index 2>&1 >nul

REM 装 onnxruntime-gpu (需要先装其依赖)
pip install "%PKGS%\coloredlogs-15*" "%PKGS%\humanfriendly-10*" "%PKGS%\pyreadline3-3*"^
 "%PKGS%\flatbuffers-25*" "%PKGS%\packaging-26*" "%PKGS%\protobuf-7*"^
 --no-index 2>&1 >nul
pip install "%PKGS%\onnxruntime_gpu-1.21*" --no-index 2>&1 >nul

REM 装 ultralytics 及其依赖 (用 --no-deps 手动控制)
pip install "%PKGS%\numpy-2.4*" "%PKGS%\pillow-12*" "%PKGS%\opencv_python-4.10*"^
 "%PKGS%\pyyaml-6.0*" "%PKGS%\requests-2.34*" "%PKGS%\matplotlib-3.11*"^
 "%PKGS%\pandas-3.0*" "%PKGS%\scipy-1.17*" "%PKGS%\seaborn-0.13*"^
 "%PKGS%\tqdm-4.70*" "%PKGS%\psutil-7.2*"^
 --no-index 2>&1 >nul

REM matplotlib & pandas deps
pip install "%PKGS%\contourpy-1.3*" "%PKGS%\cycler-0.12*" "%PKGS%\fonttools-4.63*"^
 "%PKGS%\kiwisolver-1.5*" "%PKGS%\pyparsing-3.3*" "%PKGS%\python_dateutil-2.9*"^
 "%PKGS%\six-1.17*" "%PKGS%\colorama-0.4*" "%PKGS%\tzdata-2026*"^
 --no-index 2>&1 >nul

REM requests deps
pip install "%PKGS%\charset_normalizer-3.4*" "%PKGS%\idna-3.18*"^
 "%PKGS%\urllib3-2.7*" "%PKGS%\certifi-2026*" --no-index 2>&1 >nul

REM ultralytics
pip install "%PKGS%\nvidia_ml_py-13*" "%PKGS%\polars-1.43*"^
 "%PKGS%\polars_runtime_32-1.43*" "%PKGS%\ml_dtypes-0.5*"^
 "%PKGS%\ultralytics_thop-2.1*" "%PKGS%\thop-0.1*"^
 "%PKGS%\mpmath-1.4*" "%PKGS%\markupsafe-3.0*"^
 --no-index 2>&1 >nul
pip install "%PKGS%\ultralytics-8.4.83*" --no-index --no-deps 2>&1 >nul

REM ONNX tools
pip install "%PKGS%\onnx-1.16*" "%PKGS%\onnxslim-0.1*"^
 --no-index 2>&1 >nul

REM pywin32 (post-install)
pip install "%PKGS%\pywin32-312*" --no-index 2>&1 >nul
python "%VENV%\Scripts\pywin32_postinstall.py" -install -silent 2>&1 >nul

echo   ✅ 所有依赖安装完成 (离线)
echo.

REM ==========================================
REM Step 6: Export ONNX model
REM ==========================================
echo [6/6] 导出 ONNX 模型...

REM Fix CUDA PATH (覆盖系统 CUDA, 使用 PyTorch CUDA 12.8)
for /f "tokens=*" %%i in ('python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))" 2^>nul') do set "TORCH_LIB=%%i"
if exist "%TORCH_LIB%\cublas64_12.dll" (
    set "PATH=%TORCH_LIB%;%PATH%"
    set "CUDA_PATH=%TORCH_LIB%\.."
)

if exist "%ROOT%models\cs2_best.onnx" (
    echo   ✅ ONNX 模型已存在, 跳过导出
) else (
    echo   正在导出... (可能需要 1-2 分钟)
    cd /d "%ROOT%"
    python scripts\export_onnx.py 2>&1
    if exist "models\cs2_best.onnx" (
        echo   ✅ ONNX 模型导出成功
    ) else (
        echo   ⚠️  ONNX 导出失败, 请检查: python scripts\export_onnx.py
    )
)
echo.

REM ==========================================
REM Verify
REM ==========================================
echo ╔══════════════════════════════════════════╗
echo ║  验证安装...                              ║
echo ╚══════════════════════════════════════════╝
python -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}  GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul
python -c "import onnxruntime; print(f'  ONNX Runtime {onnxruntime.__version__}  Providers={onnxruntime.get_available_providers()}')" 2>nul
python -c "import cv2; print(f'  OpenCV {cv2.__version__}')" 2>nul
python -c "import ultralytics; print(f'  Ultralytics {ultralytics.__version__}')" 2>nul

echo.
for %%f in (models\cs2_best.onnx models\best.pt) do (
    if exist "%ROOT%%%f" (
        for %%s in ("%ROOT%%%f") do echo   ✅ %%~nxf  %%~zs bytes
    )
)

echo.
echo ╔══════════════════════════════════════════╗
echo ║  ✅ 部署完成！                            ║
echo ╠══════════════════════════════════════════╣
echo ║  双击 "开始运行.bat" 启动推理              ║
echo ║  双击 "开始运行+自瞄.bat" 启动+自动瞄准     ║
echo ╚══════════════════════════════════════════╝
echo.
pause
