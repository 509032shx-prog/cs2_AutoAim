# -*- coding: utf-8 -*-
"""游戏内 HUD 叠加层 - tkinter 实现（无 OpenCV 窗口）"""
import tkinter as tk
import threading
import time


class GameHUD:
    def __init__(self):
        self.root = None
        self.label = None
        self.running = False
    
    def start(self):
        self.running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        time.sleep(0.3)  # 等窗口创建
    
    def _run(self):
        self.root = tk.Tk()
        self.root.title("HUD")
        self.root.geometry("300x80+10+10")
        self.root.overrideredirect(True)          # 无边框
        self.root.attributes('-topmost', True)     # 置顶
        self.root.attributes('-alpha', 0.75)       # 半透明
        self.root.configure(bg='#0a0a0a')
        
        self.label = tk.Label(
            self.root, text="Loading...", fg='#3cff3c', bg='#0a0a0a',
            font=('Consolas', 10), justify='left', anchor='w', padx=6, pady=4
        )
        self.label.pack(fill='both', expand=True)
        
        # 定时刷新（让窗口响应）
        def tick():
            if self.running:
                self.root.after(50, tick)
        self.root.after(50, tick)
        self.root.mainloop()
    
    def update(self, lines):
        if self.label and self.root:
            text = '\n'.join(lines[:6])
            self.root.after(0, lambda: self.label.config(text=text))
    
    def stop(self):
        self.running = False
        if self.root:
            self.root.after(0, self.root.destroy)


_hud = None

def hud_start():
    global _hud
    _hud = GameHUD()
    _hud.start()

def hud_update(lines):
    if _hud:
        _hud.update(lines)

def hud_stop():
    global _hud
    if _hud:
        _hud.stop()
        _hud = None
