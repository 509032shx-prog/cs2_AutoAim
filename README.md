# CS2 AutoAim - ONNX 推理引擎

基于 YOLOv26s 的 CS2 玩家头部检测 + 自瞄系统, 使用 ONNX Runtime 高速推理。

---

## 🚀 两种部署方式

### 🎯 方式一：解压即用（推荐，给朋友/测试）

[📥 下载便携版 (v1.0.0)](https://github.com/509032shx-prog/cs2_AutoAim/releases/download/v1.0.0/cs2_AotoAim_便携版.7z)

```
1. 下载上面的 .7z 文件（约 1.9GB）
2. 解压到任意目录
3. 双击「开始运行.bat」或「开始运行+自瞄.bat」
4. 开干 ✅
```

> 无需安装 Python、CUDA Toolkit、任何依赖。一站式打包。

### 🔧 方式二：从源码安装（开发者）

```
1. git clone https://github.com/509032shx-prog/cs2_AutoAim.git
2. 双击「安装环境.bat」（需要联网, 自动从 PyPI 安装全部依赖）
3. 双击「开始运行.bat」或「开始运行+自瞄.bat」
```

> 需要: Python 3.10~3.12 + NVIDIA 驱动 ≥525

---

## 📊 性能对比

| 格式 | 推理速度 | 训练 | 体积 | 部署依赖 |
|------|----------|------|------|----------|
| `.pt` (PyTorch) | 基准 | ✅ 可训练 | ~19 MB | PyTorch + CUDA Toolkit |
| `.onnx` (ONNX) | **1.3~3x 更快** | ❌ 不可训练 | ~18 MB | onnxruntime-gpu (自带 CUDA) |
| `.engine` (TensorRT) | 最快 (2~4x) | ❌ 不可训练 | ~15 MB | TensorRT (需编译) |

---

## 📁 项目结构

```
cs2_AutoAim\
├── 安装环境.bat              ← 在线安装 (克隆后运行)
├── 开始运行.bat              ← 双击启动推理 (侧键触发自瞄)
├── 开始运行+自瞄.bat          ← 双击启动推理 (F3切换持续自瞄)
├── 一键部署.bat              ← 完整离线部署 (需离线包)
├── models\
│   ├── best.pt               (PyTorch 训练好的模型 v4, mAP50=0.789)
│   └── cs2_best.onnx         (ONNX FP16, end2end)
├── src\
│   ├── inference_onnx.py      (ONNX Runtime 推理)
│   ├── inference_pt.py        (PyTorch 备用推理)
│   ├── screen_capture.py      (屏幕/窗口捕获)
│   └── mouse_control.py       (鼠标控制)
├── scripts\
│   ├── export_onnx.bat        (ONNX 导出脚本)
│   └── export_onnx.py
├── requirements.txt
└── README.md
```

---

## 🎮 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window "CS2"` | 全屏 | 指定游戏窗口名 |
| `--monitor 1` | 0 | 指定显示器 |
| `--conf 0.5` | 0.25 | 置信度阈值 |
| `--aimbot` | 关 | 启用持续自瞄 |
| `--aim-speed 1.5` | 1.0 | 鼠标移动速度 |
| `--aim-smoothing 2.5` | 2.0 | 平滑系数 |

## ⌨️ 运行时控制

- **按住鼠标侧后键** — 自瞄（松开停止）
- **F3** — 切换持续自瞄模式
- **Q** — 退出

---

## 🛡️ 环境隔离

即使电脑已有其他版本 CUDA、PyTorch、conda，也**不会冲突**：

| 隔离层 | 机制 |
|--------|------|
| Python 包 | `venv\` 虚拟环境，与系统 Python/conda 完全隔离 |
| PyTorch | 安装在 venv 内 (CUDA 12.8)，不碰系统 PyTorch |
| CUDA DLLs | 启动时自动将 `torch\lib\` 加到 PATH 最前面 |

> ⚠️ **唯一前提**: 电脑有 NVIDIA 驱动 ≥525。**不需要装 CUDA Toolkit**。

---

## 📝 模型信息

- **架构**: YOLOv26s
- **类别**: person (人), head (头)
- **输入尺寸**: 640x640
- **精度**: FP16
- **mAP50**: 0.789 (v4)
- **训练数据**: CS2 Merged v4 (16,423 images)

---

## ⚠️ 常见问题

**Q: 提示"虚拟环境不存在"**
→ 先运行「安装环境.bat」（在线）或「一键部署.bat」（离线）

**Q: 推理用的是 CPU 不是 GPU**
→ 检查 NVIDIA 驱动是否安装; 启动日志看是否显示 CUDAExecutionProvider

**Q: 自瞄不准**
→ 调整参数: `--aim-smoothing 1.5 --aim-min-conf 0.5`
