"""從句子裡抓亮度數字。模型只判斷「這是 SET_BRIGHTNESS」，數字由這裡負責。

⚠️ ESP32 的 C++ 版要跟這裡邏輯一致。
"""
import re

DIGITS = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9}


def chinese_to_int(s):
    """中文數字 → 整數，支援 0~100。例：四十五 → 45、十 → 10、一百 → 100"""
    if s.endswith("百"):
        return DIGITS.get(s[:-1], 1) * 100
    if "十" in s:
        tens, _, ones = s.partition("十")
        return DIGITS.get(tens, 1) * 10 + DIGITS.get(ones, 0)   # 「十」前面沒字 = 一十
    return DIGITS.get(s)


def extract_number(text):
    """回傳句子裡的第一個數字（限制在 0~100），找不到回傳 None"""
    text = text.replace("百分之", "")                    # 不然「百分之四十五」的「百」會被當成 100
    m = re.search(r"\d+", text)                        # 阿拉伯數字：80、100
    if m:
        n = int(m.group())
    else:
        m = re.search(r"[零一二兩三四五六七八九十百]+", text)  # 中文數字：四十五
        n = chinese_to_int(m.group()) if m else None
    return None if n is None else max(0, min(100, n))


if __name__ == "__main__":
    for s in ["亮度調到80", "燈調到15%", "亮度百分之四十五", "調成一百", "亮度十", "開燈"]:
        print(f"{s:<12} → {extract_number(s)}")
