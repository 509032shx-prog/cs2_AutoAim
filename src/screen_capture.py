# -*- coding: utf-8 -*-
"""
Screen / Window Capture Utilities (DXCam - D3D11)
====================================================
D3D11 capture → ~240 FPS, ~0.5ms grab vs ImageGrab's 50-200ms
"""

import numpy as np
import cv2
import win32gui
import win32con
from typing import Optional, Tuple, Generator

import dxcam


def get_screen(region: Optional[Tuple[int, int, int, int]] = None) -> Generator[np.ndarray, None, None]:
    """
    DXCam 屏幕捕获
    Args:
        region: (left, top, right, bottom) 或 None(全屏)
    Yields:
        BGR numpy array (H, W, 3)
    """
    camera = dxcam.create(output_idx=0, output_color="BGR")
    if camera is None:
        raise RuntimeError("DXCam 初始化失败, 检查 D3D11 驱动")

    if region:
        left, top, right, bottom = region
        camera.start(region=(left, top, right, bottom), target_fps=240, video_mode=True)
    else:
        camera.start(target_fps=240, video_mode=True)

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is not None:
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                yield frame
    finally:
        camera.stop()


def get_window(window_name: str) -> Generator[np.ndarray, None, None]:
    """
    捕获指定窗口 (DXCam)

    优化策略:
    - 每 60 帧更新一次窗口位置 (应对窗口拖动)
    - 窗口最小化时 fallback 全屏
    """
    def enum_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if window_name.lower() in title.lower():
                windows.append(hwnd)

    windows = []
    win32gui.EnumWindows(enum_callback, windows)

    if not windows:
        print(f"⚠️  找不到包含 '{window_name}' 的窗口, 使用全屏")
        yield from get_screen()
        return

    hwnd = windows[0]
    title = win32gui.GetWindowText(hwnd)
    print(f"✅ 找到窗口: {title} (hwnd={hwnd})")

    camera = dxcam.create(output_idx=0, output_color="BGR")
    if camera is None:
        print("⚠️  DXCam 失败, 回退 PIL (慢)")
        # Fallback: use PIL ImageGrab
        from PIL import ImageGrab
        while True:
            try:
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                if right - left > 0 and bottom - top > 0:
                    img = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)
                    yield cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            except Exception:
                yield from get_screen()
                return
        return

    frame_idx = 0
    camera.start(target_fps=240, video_mode=True)

    try:
        while True:
            # 每 60 帧刷新窗口位置
            if frame_idx % 60 == 0:
                try:
                    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                    w, h = right - left, bottom - top
                    if w <= 0 or h <= 0:
                        raise ValueError("Window minimized")
                    camera.stop()
                    camera.start(region=(left, top, right, bottom), target_fps=240, video_mode=True)
                except Exception:
                    # 窗口失效, 切全屏
                    camera.stop()
                    yield from get_screen()
                    return

            frame = camera.get_latest_frame()
            if frame is not None:
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                yield frame

            frame_idx += 1
    finally:
        camera.stop()


def get_monitor(monitor_index: int = 0) -> Generator[np.ndarray, None, None]:
    """捕获指定显示器 (DXCam)"""
    camera = dxcam.create(output_idx=monitor_index, output_color="BGR")
    if camera is None:
        raise RuntimeError(f"DXCam 初始化显示器 {monitor_index} 失败")

    print(f"✅ 显示器 {monitor_index}: DXCam D3D11 模式 ({camera.width}x{camera.height})")
    camera.start(target_fps=240, video_mode=True)

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is not None:
                # 确保 BGR 3 通道 (dxcam 可能返回 BGRA)
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                yield frame
    finally:
        camera.stop()


if __name__ == "__main__":
    """Test"""
    print("测试 DXCam 屏幕捕获")
    print("1: 全屏")
    print("2: 窗口")
    print("3: 显示器")

    choice = input("选择: ").strip()

    if choice == "2":
        wname = input("窗口名: ").strip()
        gen = get_window(wname)
    elif choice == "3":
        midx = int(input("显示器索引: ").strip())
        gen = get_monitor(midx)
    else:
        gen = get_screen()

    for frame in gen:
        h, w = frame.shape[:2]
        scale = min(1.0, 1280 / w)
        display = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cv2.putText(display, f"{w}x{h} DXCam", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Screen Capture - DXCam", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
