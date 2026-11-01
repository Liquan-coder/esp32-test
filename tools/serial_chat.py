"""跟 ESP32 聊天：在電腦上打完整句（選字、Backspace 都在本機處理），按 Enter 才整句送出

用法：uv run python tools/serial_chat.py
"""
import glob
import time

import serial

port = glob.glob("/dev/cu.usbmodem*")[0]           # 自動找板子的序列埠
ser = serial.Serial(port, 115200, timeout=0.1)
print(f"已連線 {port}，輸入 q 離開")

while True:
    text = input("> ")                              # 本機輸入框：輸入法、Backspace 都正常
    if text == "q":
        break
    ser.write((text + "\n").encode("utf-8"))        # 字串 → UTF-8 bytes，一次送出
    time.sleep(0.2)                                 # 等板子回話
    while line := ser.readline():
        print(line.decode("utf-8", errors="replace").rstrip())
