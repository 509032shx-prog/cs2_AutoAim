# -*- coding: utf-8 -*-
"""
ONNX Runtime Inference for CS2 Aimbot
======================================
使用 onnxruntime-gpu 进行高速推理。
onnxruntime-gpu 自带 CUDA 12.x + cuDNN, 无需单独安装 CUDA Toolkit。

用法:
  python src/inference_onnx.py                    # 屏幕捕获模式
  python src/inference_onnx.py --source test.jpg   # 单张图片
  python src/inference_onnx.py --source video.mp4  # 视频
  python src/inference_onnx.py --window "Counter-Strike"  # 指定窗口
  python src/inference_onnx.py --aimbot            # 启用自瞄
"""

import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import onnxruntime as ort
import win32api

from src.screen_capture import get_screen, get_window, get_monitor
from src.hud import hud_start, hud_update, hud_stop

# 鼠标侧键虚拟键码
VK_XBUTTON1 = 0x05   # 侧后键 (鼠标4键)
VK_XBUTTON2 = 0x06   # 侧前键 (鼠标5键)

# 键盘虚拟键码
VK_Q = 0x51
VK_F3 = 0x72
VK_F8 = 0x77


class ONNXDetector:
    """
    YOLO 检测器 (ONNX Runtime)
    
    模型格式: end2end (含NMS), output=(1, 300, 6)
    每行: [x1, y1, x2, y2, confidence, class_id]
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf_thres: float = 0.25,
        fp32_input: bool = True,  # ONNX input is float32
    ):
        self.imgsz = imgsz
        self.conf_thres = conf_thres

        # 初始化 ONNX Runtime
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            }),
            "CPUExecutionProvider",
        ]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(model_path, opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        outputs = self.session.get_outputs()
        print(f"  ONNX input:  {self.session.get_inputs()[0].name} "
              f"shape={self.session.get_inputs()[0].shape}")
        for o in outputs:
            print(f"  ONNX output: {o.name} shape={o.shape}")

        self.warmup_done = False

    def warmup(self, n: int = 10):
        """GPU 预热"""
        if self.warmup_done:
            return
        dummy = np.random.randn(1, 3, self.imgsz, self.imgsz).astype(np.float32)
        for _ in range(n):
            self.session.run(None, {self.input_name: dummy})
        self.warmup_done = True

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        预处理 (与 ultralytics 导出一致):
        1. Letterbox resize → 640x640
        2. BGR → RGB
        3. Normalize (÷255)
        4. HWC → NCHW
        5. float32 + contiguous
        """
        h0, w0 = img.shape[:2]

        # Letterbox
        r = self.imgsz / max(h0, w0)
        if r < 1:
            new_h, new_w = int(h0 * r), int(w0 * r)
            img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_r = img
            new_h, new_w = h0, w0

        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_h = (self.imgsz - new_h) // 2
        pad_w = (self.imgsz - new_w) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_r

        self._letterbox = (pad_h, pad_w, r, h0, w0)

        # BGR → RGB
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # Normalize + HWC→CHW + batch
        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0)

        return np.ascontiguousarray(blob)

    def postprocess(self, output, orig_shape=None):
        """
        后处理: (1, 300, 6) → boxes, scores, class_ids (原始图像坐标)
        
        输出格式: [x1, y1, x2, y2, confidence, class_id]
        boxes 已经是归一化坐标 (0~640), 需要缩放回原图
        """
        pad_h, pad_w, ratio, h0, w0 = self._letterbox
        pred = output[0]  # (300, 6)

        # 过滤低置信度
        mask = pred[:, 4] >= self.conf_thres
        pred = pred[mask]

        if len(pred) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        # 缩放 box 回原始图像坐标
        boxes = pred[:, :4].copy()
        if ratio < 1:
            boxes[:, [0, 2]] -= pad_w
            boxes[:, [1, 3]] -= pad_h
            boxes /= ratio

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

        scores = pred[:, 4]
        class_ids = pred[:, 5].astype(int)

        return boxes, scores, class_ids

    def detect(self, img: np.ndarray):
        """
        执行检测
        Returns: boxes (N,4), scores (N,), class_ids (N,)
        """
        blob = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})
        return self.postprocess(outputs[0])

    def detect_timed(self, img: np.ndarray):
        """检测 + 耗时统计"""
        t0 = time.perf_counter()
        blob = self.preprocess(img)
        t_prep = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: blob})
        t_inf = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        boxes, scores, class_ids = self.postprocess(outputs[0])
        t_post = (time.perf_counter() - t2) * 1000

        return boxes, scores, class_ids, (t_prep, t_inf, t_post)


# ============================================================
# 绘制函数
# ============================================================
def draw_detections(frame, boxes, scores, class_ids, names,
                    show_aim_line=True, crosshair_pos=None):
    h, w = frame.shape[:2]
    if crosshair_pos is None:
        crosshair_pos = (w // 2, h // 2)
    cx_s, cy_s = crosshair_pos

    for box, score, cls in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        label = f"{names.get(int(cls), 'obj')} {score:.2f}"
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if 'head' in names.get(int(cls), '').lower():
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        if show_aim_line and 'head' in names.get(int(cls), '').lower():
            cv2.line(frame, (cx_s, cy_s), (cx, cy),
                     (255, 255, 0), 1, cv2.LINE_AA)

    cv2.drawMarker(frame, (cx_s, cy_s), (0, 255, 255),
                   cv2.MARKER_CROSS, 25, 1)
    return frame


def get_aim_target(boxes, scores, class_ids, names, screen_center, min_conf=0.3):
    """获取最近的自瞄目标 (head)"""
    cx_s, cy_s = screen_center
    best_dist = float('inf')
    best_target = None

    for box, score, cls in zip(boxes, scores, class_ids):
        if score < min_conf:
            continue
        if 'head' not in names.get(int(cls), '').lower():
            continue

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist = ((cx - cx_s) ** 2 + (cy - cy_s) ** 2) ** 0.5

        if dist < best_dist:
            best_dist = dist
            best_target = (cx, cy)

    return best_target


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="CS2 Aimbot ONNX Inference")
    parser.add_argument("--model", type=str,
                        default=str(PROJECT_ROOT / "models" / "cs2_best.onnx"))
    parser.add_argument("--source", type=str, default="screen")
    parser.add_argument("--window", type=str, default=None)
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--aimbot", action="store_true")
    parser.add_argument("--aim-speed", type=float, default=1.0)
    parser.add_argument("--aim-smoothing", type=float, default=2.0)
    parser.add_argument("--aim-min-conf", type=float, default=0.4)
    parser.add_argument("--show-fps", action="store_true", default=True)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("CS2 Aimbot - ONNX Runtime Inference")
    print("=" * 60)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 模型不存在: {model_path}")
        sys.exit(1)

    detector = ONNXDetector(str(model_path), imgsz=args.img_size)
    print(f"🎯 执行引擎: {detector.session.get_providers()}")

    if not args.no_warmup:
        print("🔥 GPU 预热中...")
        detector.warmup(10)
        print("✅ 预热完成\n")

    names = {0: "person", 1: "head"}

    # 始终初始化自瞄控制器 (鼠标侧键触发)
    from src.mouse_control import MouseController
    aim_controller = MouseController(
        smoothing=args.aim_smoothing, speed=args.aim_speed)
    aim_always_on = args.aimbot  # --aimbot 持续模式

    if args.source == "screen":
        if args.window:
            frame_gen = get_window(args.window)
        else:
            frame_gen = get_monitor(args.monitor)

        hud_start()
        time.sleep(0.3)

        fps_history = []
        prev_time = time.perf_counter()
        t_start = time.perf_counter()
        last_hud = 0
        frame_count = 0
        f3_prev = False

        print("🎮 开始推理")
        print("   按住 [鼠标侧后键] = 自瞄 (松开停止)")
        print("   Q=退出  F3=切换持续自瞄  F8=退出")
        if aim_always_on:
            print("   🎯 持续自瞄模式已开启")
        print()
        for frame in frame_gen:
            frame_count += 1
            t_loop_start = time.perf_counter()

            t0 = time.perf_counter()
            boxes, scores, class_ids, (t_prep, t_inf, t_post) = detector.detect_timed(frame)
            t_total = (time.perf_counter() - t0) * 1000

            h, w = frame.shape[:2]
            crosshair = (w // 2, h // 2)

            # 自瞄触发: 鼠标侧后键按住 或 持续模式
            side_held = win32api.GetAsyncKeyState(VK_XBUTTON1) < 0
            if aim_always_on or side_held:
                target = get_aim_target(boxes, scores, class_ids, names,
                                        crosshair, args.aim_min_conf)
                if target:
                    dx = target[0] - crosshair[0]
                    dy = target[1] - crosshair[1]
                    aim_controller.move_smooth(dx, dy)

            # --- 实时 FPS 计算 ---
            now = time.perf_counter()
            dt = (now - prev_time) * 1000  # ms since last frame
            prev_time = now
            fps = 1000.0 / dt if dt > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 100:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / max(len(fps_history), 1)

            # 每帧打印到控制台 (前 5 帧)
            if frame_count <= 5:
                print(f"[F#{frame_count}] dt={dt:.1f}ms  fps={fps:.0f}  prep={t_prep:.1f}ms  inf={t_inf:.1f}ms  post={t_post:.1f}ms  total={t_total:.1f}ms")

            # --- HUD 更新 (每 100ms) ---
            if now - last_hud > 0.1:
                n_heads = sum(1 for c in class_ids if c == 1)
                n_persons = sum(1 for c in class_ids if c == 0)

                if side_held and not aim_always_on:
                    aim_status = "▶ AIMING"
                elif aim_always_on:
                    aim_status = "◉ AUTO AIM"
                else:
                    aim_status = "○ idle"

                hud_update([
                    f"FPS: {avg_fps:5.0f} | dt: {dt:5.1f}ms | {aim_status}",
                    f"Prep:{t_prep:5.1f} Inf:{t_inf:4.1f} Post:{t_post:4.1f}ms",
                    f"Total:{t_total:5.1f}ms | {n_persons}P + {n_heads}H",
                ])
                last_hud = now

            # --- 键盘检测 ---
            if win32api.GetAsyncKeyState(VK_Q) < 0 or win32api.GetAsyncKeyState(VK_F8) < 0:
                break

            f3_now = win32api.GetAsyncKeyState(VK_F3) < 0
            if f3_now and not f3_prev:
                aim_always_on = not aim_always_on
                aim_controller.reset()
                print(f"  F3 持续自瞄: {'ON' if aim_always_on else 'OFF (鼠标侧键触发)'}")
            f3_prev = f3_now

        hud_stop()

    elif args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"📷 摄像头 {args.source} (Q=退出)")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            boxes, scores, class_ids = detector.detect(frame)
            t_infer = (time.perf_counter() - t0) * 1000
            frame = draw_detections(frame, boxes, scores, class_ids, names)
            cv2.putText(frame, f"Infer: {t_infer:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("CS2 Aimbot - Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif any(args.source.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.bmp')):
        print(f"📁 图片: {args.source}")
        frame = cv2.imread(args.source)
        t0 = time.perf_counter()
        boxes, scores, class_ids = detector.detect(frame)
        t_infer = (time.perf_counter() - t0) * 1000
        print(f"  推理: {t_infer:.1f}ms, 检测: {len(boxes)} 个目标")

        frame = draw_detections(frame, boxes, scores, class_ids, names)
        cv2.putText(frame, f"Infer: {t_infer:.1f}ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out_path = PROJECT_ROOT / "models" / f"_result_{Path(args.source).name}"
        cv2.imwrite(str(out_path), frame)
        print(f"✅ 结果: {out_path}")

        scale = min(1.0, 1280 / frame.shape[1])
        show = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        cv2.imshow("CS2 Aimbot - Result", show)
        print("按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print(f"🎬 视频: {args.source}")
        cap = cv2.VideoCapture(args.source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            boxes, scores, class_ids = detector.detect(frame)
            frame = draw_detections(frame, boxes, scores, class_ids, names)
            cv2.putText(frame, f"Infer: {(time.perf_counter()-t0)*1000:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("CS2 Aimbot - Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    print("👋 推理结束")


if __name__ == "__main__":
    main()
