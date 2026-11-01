# 模型輸出的唯一定義來源。C++ 的 enum 之後會從這裡自動產生，改這裡就好。

# 第 1 題：要做什麼？
INTENTS = [
    "NONE",            # 跟燈無關，或是否定句（「不要開燈」）
    "ON",
    "OFF",
    "BRIGHTER",
    "DIMMER",
    "SET_BRIGHTNESS",  # 數字由 regex 抓，模型只判斷「這是設定亮度」
    "SET_COLOR",
    "BLINK",
]

# 第 2 題：什麼顏色？（句子沒提到顏色就是 NONE）
COLORS = ["NONE", "RED", "GREEN", "BLUE", "YELLOW", "PURPLE", "WHITE"]
