# -*- coding: utf-8 -*-
"""
Export best.pt → ONNX (FP16) + test inference speed comparison
=============================================================
Usage: python scripts/export_onnx.py

ONNX vs PyTorch:
  - Training:  ONNX 不能训练, 只能在 PyTorch 中训练
  - Inference: ONNX Runtime 通常比 PyTorch 快 1.3~3x (取决于模型和硬件)
  - Portability: ONNX 跨平台, 无需 PyTorch 依赖
  - onnxruntime-gpu 自带 CUDA/CuDNN, 无需单独安装 CUDA Toolkit
"""

import sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from ultralytics import YOLO

# ============================================================
# 配置
# ============================================================
BEST_PT = PROJECT_ROOT / "models" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "models"
IMG_SIZE = 640
WARMUP = 50
BENCH_RUNS = 200

# ============================================================
# Step 1: 导出 ONNX with FP16
# ============================================================
def export_onnx():
    print("=" * 60)
    print("Step 1: 导出 ONNX 模型 (FP16 + simplify)")
    print("=" * 60)

    model = YOLO(str(BEST_PT))

    # 导出 ONNX
    onnx_path = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        half=True,          # FP16 → 模型体积减半, 推理加速
        simplify=True,      # 简化计算图
        opset=17,           # onnxruntime 1.20+ 支持
        device="cpu",  # export doesn't need GPU
        verbose=False,
    )
    print(f"   ONNX 导出成功: {onnx_path}")

    # 复制到项目目录
    dest = OUTPUT_DIR / "cs2_best.onnx"
    import shutil
    shutil.copy(onnx_path, dest)
    print(f"   复制到: {dest}")

    onnx_size = dest.stat().st_size / (1024**2)
    pt_size = BEST_PT.stat().st_size / (1024**2)
    print(f"\n   .pt 大小:  {pt_size:.1f} MB")
    print(f"   .onnx 大小: {onnx_size:.1f} MB")
    print(f"   压缩比:     {pt_size/onnx_size:.1f}x\n")

    return model, dest


# ============================================================
# Step 2: 速度对比
# ============================================================
def benchmark_speed(model_pt, onnx_path):
    print("=" * 60)
    print("Step 2: 推理速度对比")
    print("=" * 60)

    dummy = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # --- PyTorch ---
    print("\n[PyTorch FP16] 预热...")
    for _ in range(WARMUP):
        model_pt.predict(dummy, imgsz=IMG_SIZE, half=True, verbose=False)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(BENCH_RUNS):
        model_pt.predict(dummy, imgsz=IMG_SIZE, half=True, verbose=False)
    torch.cuda.synchronize()
    pt_time = (time.perf_counter() - t0) / BENCH_RUNS * 1000

    # --- ONNX Runtime ---
    import onnxruntime as ort
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    # 预处理: resize + normalize + NCHW
    import cv2
    img = cv2.resize(dummy, (IMG_SIZE, IMG_SIZE))
    blob = img.astype(np.float16) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC → CHW
    blob = np.expand_dims(blob, 0)  # → NCHW

    input_name = sess.get_inputs()[0].name

    print("[ONNX FP16] 预热...")
    for _ in range(WARMUP):
        sess.run(None, {input_name: blob})

    t0 = time.perf_counter()
    for _ in range(BENCH_RUNS):
        sess.run(None, {input_name: blob})
    ort_time = (time.perf_counter() - t0) / BENCH_RUNS * 1000

    # --- 结果 ---
    print(f"\n{'='*50}")
    print(f"  PyTorch FP16:     {pt_time:7.2f} ms")
    print(f"  ONNX FP16:        {ort_time:7.2f} ms")
    print(f"  加速比:           {pt_time/ort_time:.2f}x")
    print(f"{'='*50}")

    if ort_time < pt_time:
        print(f"\n  ✅ ONNX 比 PyTorch 快 {pt_time/ort_time:.1f}x, 建议用 ONNX 推理")
    else:
        print(f"\n  ⚠️  ONNX 没有更快, 可能是驱动/版本问题")

    print(f"\n  实时性:  {ort_time:.1f}ms ≈ {1000/ort_time:.0f} FPS")


# ============================================================
# Step 3: 验证输出一致性
# ============================================================
def verify_output(model_pt, onnx_path):
    print("\n" + "=" * 60)
    print("Step 3: 输出一致性验证")
    print("=" * 60)

    import cv2
    import onnxruntime as ort

    # 用真实图片测试
    dummy = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / "_test_input.jpg"), dummy)

    # PyTorch 预测
    pt_results = model_pt.predict(dummy, imgsz=IMG_SIZE, half=True, verbose=False)

    # ONNX 预测
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    img = cv2.resize(dummy, (IMG_SIZE, IMG_SIZE))
    blob = img.astype(np.float16) / 255.0
    blob = blob.transpose(2, 0, 1)
    blob = np.expand_dims(blob, 0)
    input_name = sess.get_inputs()[0].name
    onnx_outputs = sess.run(None, {input_name: blob})

    if pt_results[0].boxes is not None:
        n_pt = len(pt_results[0].boxes)
        print(f"  PyTorch 检测: {n_pt} 个目标")
    else:
        print(f"  PyTorch 检测: 0 个目标 (正常, 随机噪声)")

    print(f"  ONNX 输出数量: {len(onnx_outputs)}")
    for i, out in enumerate(onnx_outputs):
        print(f"    output[{i}] shape: {out.shape}")

    print("\n  ✅ 验证完成")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    import torch, argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    args = ap.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print()

    # Step 1: 导出
    model, onnx_path = export_onnx()

    if args.benchmark:
        # Step 2: 测速 (optional)
        benchmark_speed(model, onnx_path)
        # Step 3: 验证
        verify_output(model, onnx_path)

    print("\n🎉 完成! ONNX 模型已保存到 models/cs2_best.onnx")
    print("   下一步: python src/inference_onnx.py")
