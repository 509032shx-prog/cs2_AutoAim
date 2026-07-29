# -*- coding: utf-8 -*-
"""
Screen / Window Capture Utilities
==================================
Supports: full screen, specific window, monitor region
"""

import numpy as np
import cv2
from PIL import ImageGrab
import win32gui
import win32con
from typing import Optional, Tuple, Generator


def get_screen(region: Optional[Tuple[int, int, int, int]] = None) -> Generator[np.ndarray, None, None]:
    """
    实时捕获屏幕
    Args:
        region: (left, top, right, bottom) 或 None(全屏)
    Yields:
        BGR numpy array (H, W, 3)
    """
    while True:
        img = ImageGrab.grab(bbox=region, all_screens=True)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        yield frame


def get_window(window_name: str) -> Generator[np.ndarray, None, None]:
    """
    捕获指定窗口
    Args:
        window_name: 窗口标题 (支持部分匹配)
    Yields:
        BGR numpy array (H, W, 3)
    """
    # 先查找窗口
    def enum_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if window_name.lower() in title.lower():
                windows.append((hwnd, title))

    windows = []
    win32gui.EnumWindows(enum_callback, windows)

    if not windows:
        print(f"⚠️  找不到包含 '{window_name}' 的窗口, 使用全屏")
        yield from get_screen()
        return

    hwnd, title = windows[0]
    print(f"✅ 找到窗口: {title} (hwnd={hwnd})")

    while True:
        try:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            
            # 跳过最小化窗口
            if right - left <= 0 or bottom - top <= 0:
                yield from get_screen()
                return

            img = ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            yield frame
        except Exception as e:
            print(f"⚠️  窗口捕获异常: {e}, 切换到全屏")
            yield from get_screen()
            return


def get_monitor(monitor_index: int = 0) -> Generator[np.ndarray, None, None]:
    """
    捕获指定显示器
    Args:
        monitor_index: 显示器索引 (0=主显示器)
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32

    monitors = []

    def monitor_enum_proc(hMonitor, hdcMonitor, lprcMonitor, dwData):
        rect = lprcMonitor.contents
        monitors.append((rect.left, rect.top, rect.right, rect.bottom))
        return True

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_ulonglong,
        ctypes.c_ulonglong,
        ctypes.POINTER(wintypes.RECT),
        ctypes.c_double,
    )
    user32.EnumDisplayMonitors(0, 0, MonitorEnumProc(monitor_enum_proc), 0)

    if monitor_index >= len(monitors):
        print(f"⚠️  显示器 {monitor_index} 不存在, 使用主显示器")
        monitor_index = 0

    region = monitors[monitor_index]
    print(f"✅ 显示器 {monitor_index}: {region[2]-region[0]}x{region[3]-region[1]} "
          f"@ ({region[0]}, {region[1]})")
    yield from get_screen(region=region)


if __name__ == "__main__":
    """Test screen capture"""
    print("测试屏幕捕获 (按 Q 退出)")
    print("1: 全屏")
    print("2: 窗口 (输入窗口名)")
    print("3: 指定显示器")

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
        # 限制显示大小
        scale = min(1.0, 1280 / w)
        display = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cv2.putText(display, f"{w}x{h}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Screen Capture", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
