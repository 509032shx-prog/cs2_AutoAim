# CS2 AutoAim - ONNX 推理引擎

基于 YOLOv26s 的 CS2 玩家头部检测 + 自瞄系统，ONNX Runtime 高速推理。

---

## ⚡ 实测性能

| 显卡 | 分辨率 | 截图方式 | 帧率 | 预处理 | 推理 |
|------|--------|----------|------|--------|------|
| RTX 5060 Ti | 4K (3840×2160) | DXCam D3D11 | **~60 FPS** | 4-7ms | 6-8ms (FP16) |

- 截图：DXCam (D3D11 桌面复制) — 比 PIL ImageGrab 快 **200 倍**
- 推理：ONNX Runtime CUDA FP16 — 端到端 NMS
- 叠加：透明 HUD 绿色文字（无 OpenCV 弹窗）

---

## 🚀 两种部署方式

### 🎯 方式一：解压即用（推荐）

[📥 下载便携版 (v1.0.0)](https://github.com/509032shx-prog/cs2_AutoAim/releases/download/v1.0.0/cs2_AotoAim_.7z)

```
1. 下载 .7z 文件（约 2GB）
2. 解压到任意目录
3. 双击「开始运行.bat」
4. 开干 ✅
```

> 无需安装 Python、CUDA Toolkit。一站式打包。

### 🔧 方式二：从源码安装（开发者）

```
1. git clone https://github.com/509032shx-prog/cs2_AutoAim.git
2. 双击「安装环境.bat」（需要联网）
3. 双击「开始运行.bat」
```

---

## 📊 推理后端对比

| 后端 | 推理速度 | 额外依赖 |
|------|----------|----------|
| CUDA (FP16) | 6-8ms | 无（onnxruntime-gpu 自带） |
| TensorRT (FP16) | 2-4ms | 需安装 NVIDIA TensorRT SDK |

---

## 📁 项目结构

```
cs2_AutoAim\
├── 开始运行.bat              ← 双击启动（侧键自瞄）
├── 开始运行+自瞄.bat          ← 双击启动（F3 持续自瞄）
├── 安装环境.bat              ← 在线安装依赖
├── models\
│   ├── best.pt               (PyTorch v4, mAP50=0.789)
│   └── cs2_best.onnx         (ONNX FP16, end2end NMS)
├── src\
│   ├── inference_onnx.py      (ONNX Runtime 推理主程序)
│   ├── inference_pt.py        (PyTorch 备用推理)
│   ├── screen_capture.py      (DXCam D3D11 屏幕捕获)
│   ├── mouse_control.py       (鼠标控制 / 平滑自瞄)
│   └── hud.py                 (透明 HUD 叠加层)
├── scripts\
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
| `--no-warmup` | 关 | 跳过 GPU 预热 |
| `--no-tensorrt` | 关 | 禁用 TensorRT（用 CUDA） |
| `--img-size 480` | 640 | 输入尺寸 |

## ⌨️ 运行时控制

- **按住鼠标侧后键** — 自瞄（松开停止）
- **F3** — 切换持续自瞄模式
- **Q / F8** — 退出

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
→ 先运行「安装环境.bat」

**Q: 推理用的是 CPU 不是 GPU**
→ 检查 NVIDIA 驱动；启动日志会显示执行引擎

**Q: 自瞄不准**
→ `--aim-smoothing 1.5 --aim-min-conf 0.5`

**Q: 帧率低 (<30 FPS)**
→ 检查是否 D3D11 模式（日志会打印）；DXCam 需要 Win10+
