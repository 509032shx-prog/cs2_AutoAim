@echo off
chcp 65001 >nul
title CS2 AutoAim - Export ONNX

echo ============================================
echo   导出 ONNX 模型 + 速度对比
echo ============================================
echo.

REM Check venv
if exist "%~dp0..\venv\Scripts\activate.bat" (
    call "%~dp0..\venv\Scripts\activate.bat"
)

cd /d "%~dp0.."
python scripts/export_onnx.py

echo.
pause
