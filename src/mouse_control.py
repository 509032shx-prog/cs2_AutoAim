# -*- coding: utf-8 -*-
"""
Mouse Control for Aimbot
=========================
Smooth mouse movement with configurable speed/smoothing.
"""

import math
import time
import ctypes
from ctypes import wintypes
from typing import Tuple, Optional


# Win32 API constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

# For GetSystemMetrics
SM_CXSCREEN = 0
SM_CYSCREEN = 1


class MouseController:
    """
    Mouse control with smoothing.
    
    Usage:
        mouse = MouseController(smoothing=2.0, speed=1.0)
        mouse.move(dx, dy)           # relative movement
        mouse.move_to(x, y)          # absolute movement
        mouse.move_smooth(dx, dy)    # smoothed movement
    """

    def __init__(
        self,
        smoothing: float = 2.0,
        speed: float = 1.0,
        curve_strength: float = 0.3,
        max_move: int = 100,
    ):
        """
        Args:
            smoothing: 平滑系数 (越大越平滑, 但越慢), 推荐 1.5~3.0
            speed: 基础速度倍率
            curve_strength: 曲线强度 (0=直线, 1=最强曲线)
            max_move: 单次最大移动像素 (防止跳动)
        """
        self.smoothing = smoothing
        self.speed = speed
        self.curve_strength = curve_strength
        self.max_move = max_move

        self._prev_dx = 0
        self._prev_dy = 0
        self._screen_w = ctypes.windll.user32.GetSystemMetrics(0)
        self._screen_h = ctypes.windll.user32.GetSystemMetrics(1)

    @property
    def screen_size(self) -> Tuple[int, int]:
        return (self._screen_w, self._screen_h)

    def move(self, dx: int, dy: int):
        """立即移动 (离散, 无平滑)"""
        ctypes.windll.user32.mouse_event(
            MOUSEEVENTF_MOVE, dx, dy, 0, 0
        )
        # 也更新 prev 以便平滑计算
        self._prev_dx = dx
        self._prev_dy = dy

    def move_to(self, x: int, y: int):
        """移动到绝对屏幕坐标"""
        # 转换为 0~65535 的绝对坐标
        abs_x = int(x * 65535 / self._screen_w)
        abs_y = int(y * 65535 / self._screen_h)
        ctypes.windll.user32.mouse_event(
            MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
            abs_x, abs_y, 0, 0
        )

    def move_smooth(self, target_dx: float, target_dy: float):
        """
        平滑移动鼠标 (模拟人手移动轨迹)
        
        Args:
            target_dx: 目标 X 偏移
            target_dy: 目标 Y 偏移
        """
        # 计算移动距离
        distance = math.sqrt(target_dx**2 + target_dy**2)

        if distance < 1:
            return

        # 应用速度
        target_dx *= self.speed
        target_dy *= self.speed
        distance *= self.speed

        # 平滑: 与上一帧的移动做加权平均
        smooth_dx = (target_dx + self._prev_dx * (self.smoothing - 1)) / self.smoothing
        smooth_dy = (target_dy + self._prev_dy * (self.smoothing - 1)) / self.smoothing

        # 限制单次最大移动
        smooth_dx = max(-self.max_move, min(self.max_move, smooth_dx))
        smooth_dy = max(-self.max_move, min(self.max_move, smooth_dy))

        # 曲线: 添加微小偏移模拟人手晃动
        if distance > 5 and self.curve_strength > 0:
            angle = math.atan2(target_dy, target_dx)
            curve_offset = math.sin(distance * 0.1) * self.curve_strength
            smooth_dx += curve_offset * math.cos(angle + math.pi / 2)
            smooth_dy += curve_offset * math.sin(angle + math.pi / 2)

        # 执行移动
        dx_int = int(round(smooth_dx))
        dy_int = int(round(smooth_dy))

        if dx_int != 0 or dy_int != 0:
            ctypes.windll.user32.mouse_event(
                MOUSEEVENTF_MOVE, dx_int, dy_int, 0, 0
            )

        # 更新历史
        self._prev_dx = target_dx
        self._prev_dy = target_dy

    def click(self):
        """左键点击"""
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.01)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def get_cursor_pos(self) -> Tuple[int, int]:
        """获取当前鼠标位置"""
        pt = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return (pt.x, pt.y)

    def reset(self):
        """重置平滑状态"""
        self._prev_dx = 0
        self._prev_dy = 0


if __name__ == "__main__":
    print("Mouse Controller Test")
    print("=====================")
    print("Testing in 3 seconds... (move mouse to see the effect)")
    time.sleep(3)

    mouse = MouseController(smoothing=2.0, speed=0.5, curve_strength=0.3)

    print("Moving right 100px...")
    for _ in range(5):
        mouse.move_smooth(100, 0)
        time.sleep(0.016)  # ~60fps

    time.sleep(0.5)
    print("Moving down 100px...")
    for _ in range(5):
        mouse.move_smooth(0, 100)
        time.sleep(0.016)

    time.sleep(0.5)
    print("Moving diagonal with curve...")
    for _ in range(10):
        mouse.move_smooth(50, 30)
        time.sleep(0.016)

    print("Done!")
