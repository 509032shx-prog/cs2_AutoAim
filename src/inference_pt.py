# -*- coding: utf-8 -*-
"""
PyTorch Fallback Inference (backup if ONNX doesn't work)
=========================================================
Compatible with the same interface as inference_onnx.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ULTRALYTICS_ROOT = Path(r"E:\deeplearning\ultralytics-8.4.83")
sys.path.insert(0, str(ULTRALYTICS_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
from PIL import ImageGrab
import win32gui, win32con
import argparse


def main():
    parser = argparse.ArgumentParser(description="CS2 Aimbot - PyTorch Fallback")
    parser.add_argument("--model", type=str,
                        default=str(PROJECT_ROOT / "models" / "best.pt"),
                        help="PyTorch 模型路径")
    parser.add_argument("--source", type=str, default="screen")
    parser.add_argument("--window", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--show-fps", action="store_true", default=True)
    args = parser.parse_args()

    # 检查模型
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        print("   请确保 models/best.pt 存在")
        print("   或者运行: python scripts/export_onnx.py")
        sys.exit(1)

    from ultralytics import YOLO
    print(f"✅ 加载 PyTorch 模型: {model_path}")
    model = YOLO(str(model_path))

    # 屏幕捕获
    from src.screen_capture import get_screen, get_window, get_monitor

    if args.window:
        frame_gen = get_window(args.window)
    else:
        frame_gen = get_monitor(0)

    names = {0: "person", 1: "head"}
    prev_time = time.perf_counter()

    print("🎮 PyTorch 推理 (按 Q 退出)...")
    for frame in frame_gen:
        t0 = time.perf_counter()

        results = model.predict(frame, conf=args.conf, verbose=False,
                                half=True, device=0)

        t_infer = (time.perf_counter() - t0) * 1000

        h, w = frame.shape[:2]
        cx_s = w // 2
        cy_s = h // 2

        # 绘制
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            cls = results[0].boxes.cls.cpu().numpy()

            for box, score, c in zip(boxes, scores, cls):
                x1, y1, x2, y2 = map(int, box)
                label = f"{names.get(int(c), 'obj')} {score:.2f}"
                color = (0, 0, 255) if 'head' in names.get(int(c), '').lower() else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.drawMarker(frame, (cx_s, cy_s), (0, 255, 255),
                       cv2.MARKER_CROSS, 25, 1)

        if args.show_fps:
            curr = time.perf_counter()
            fps = 1.0 / (curr - prev_time)
            prev_time = curr
            cv2.putText(frame, f"FPS: {fps:.0f} | {t_infer:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        scale = min(1.0, 1280 / w)
        show = cv2.resize(frame, (int(w*scale), int(h*scale)))
        cv2.imshow("CS2 Aimbot - PyTorch", show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("👋 推理结束")


if __name__ == "__main__":
    main()
