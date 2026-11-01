"""用模板 + 隨機填空產生訓練資料 → data/train.jsonl

每行一筆：{"text": "幫我開燈", "intent": "ON", "color": "NONE"}
"""
import json
import random
from pathlib import Path

from schema import COLORS, INTENTS

random.seed(42)  # 固定亂數種子，每次產生的資料都一樣，方便比較實驗結果

# 前綴、後綴：讓同一個指令有各種口語說法
NEGATE = True   # 是否產生否定句（實驗用開關）
NEG = ["不要", "別", "不用", "先不要", "不想", "我沒有要", "我不要", "不准", "先別"]

PRE = ["", "", "", "幫我", "請", "麻煩", "可以", "欸", "嘿", "那個", "我想", "能不能"]
SUF = ["", "", "", "一下", "吧", "好嗎", "謝謝", "喔", "啦", "好不好", "嗎"]

# 顏色 → 各種中文說法
COLOR_WORDS = {
    "RED": ["紅色", "紅", "大紅色"],
    "GREEN": ["綠色", "綠", "草綠色"],
    "BLUE": ["藍色", "藍", "天藍色"],
    "YELLOW": ["黃色", "黃", "金黃色"],
    "PURPLE": ["紫色", "紫", "紫羅蘭色"],
    "WHITE": ["白色", "白", "白光"],
}

# 每個 intent 的核心句型。{c} = 顏色詞，{n} = 數字
CORE = {
    "ON": [
        "開燈", "把燈打開", "燈打開", "開一下燈", "打開燈", "燈開起來", "點燈",
        "把LED打開", "開LED", "燈亮起來", "讓燈亮", "我要開燈",
    ],
    "ON+color": [
        "開{c}燈", "把燈打開變{c}", "開燈用{c}", "打開{c}的燈", "亮{c}燈", "給我{c}的燈光",
    ],
    "OFF": [
        "關燈", "把燈關掉", "燈關掉", "關掉燈", "熄燈", "燈關起來", "把LED關掉",
        "關LED", "不用開燈了", "燈可以關了", "讓燈熄滅", "我要睡了關燈",
    ],
    "BRIGHTER": [
        "亮一點", "燈亮一點", "調亮", "把燈調亮", "再亮一點", "太暗了", "好暗",
        "看不清楚", "亮度調高", "燈開亮一點", "再亮", "亮度加大",
    ],
    "DIMMER": [
        "暗一點", "燈暗一點", "調暗", "把燈調暗", "再暗一點", "太亮了", "好刺眼",
        "眼睛好痛", "亮度調低", "燈光柔和一點", "再暗", "亮度降低",
    ],
    "SET_BRIGHTNESS": [
        "亮度調到{n}", "亮度{n}", "燈調成{n}%", "亮度設定{n}", "調到百分之{n}",
        "亮度改成{n}", "把燈調到{n}", "亮度{n}%", "燈光{n}%",
    ],
    "SET_COLOR": [
        "換成{c}", "燈變{c}", "改成{c}", "我要{c}的燈", "換{c}", "顏色改{c}",
        "燈光換成{c}", "變成{c}", "顏色調成{c}", "用{c}",
    ],
    "BLINK": [
        "閃一下", "燈閃一下", "讓燈閃爍", "一直閃", "閃爍", "燈閃起來", "閃燈",
        "讓LED閃", "燈一閃一閃", "閃個幾下",
    ],
    "BLINK+color": [
        "閃{c}燈", "{c}燈閃一下", "用{c}閃爍", "讓燈閃{c}",
    ],
}

# NONE：跟燈無關的句子 + 「看起來像但其實不是指令」的陷阱句
NONE_SENTENCES = [
    # 無關的日常句子
    "今天天氣如何", "現在幾點", "我肚子餓了", "你好", "早安", "晚安", "謝謝你",
    "今天吃什麼", "明天要上班", "播放音樂", "溫度幾度", "你叫什麼名字",
    "幫我設鬧鐘", "講個笑話", "好無聊", "我在寫程式", "外面下雨了",
    "電腦開機", "打開冷氣", "關掉電視", "把門打開", "關窗戶",
    "音量調大", "音量小一點", "風扇開強一點", "測試測試", "哈囉",
    "你在幹嘛", "等一下", "好", "不要", "嗯", "對", "算了",
    # 陷阱句：有燈相關的字，但不是控制指令
    "不要開燈", "先不要關燈", "別開燈", "燈不用開", "不要調亮",
    "燈是誰發明的", "這個燈多少錢", "我喜歡紅色", "藍色很好看",
    "LED是什麼", "燈壞了嗎", "燈好漂亮", "紅燈要停", "綠燈可以走",
    "亮度是什麼意思", "我覺得剛剛好", "燈這樣就好", "不用調了",
]


def fill(template, color_key=None):
    """把模板裡的 {c} {n} 填上隨機值"""
    text = template
    if "{c}" in text:
        text = text.replace("{c}", random.choice(COLOR_WORDS[color_key]))
    if "{n}" in text:
        text = text.replace("{n}", str(random.choice([5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100])))
    return random.choice(PRE) + text + random.choice(SUF)


def make(n_per_intent=300):
    rows = set()  # 用 set 去除重複
    for intent in INTENTS:
        if intent == "NONE":
            continue
        templates = CORE[intent]
        color_templates = CORE.get(f"{intent}+color", [])
        for _ in range(n_per_intent):
            # SET_COLOR 一定有顏色；ON/BLINK 有 40% 機率帶顏色
            if intent == "SET_COLOR" or (color_templates and random.random() < 0.4):
                color = random.choice(COLORS[1:])
                t = random.choice(templates if intent == "SET_COLOR" else color_templates)
                rows.add((fill(t, color), intent, color))
            else:
                rows.add((fill(random.choice(templates)), intent, "NONE"))

    # NONE 句子比較少，每句配上不同前後綴，擴增成多筆
    for _ in range(n_per_intent * 2):
        rows.add((fill(random.choice(NONE_SENTENCES)), "NONE", "NONE"))

    # 否定句：否定詞 + 任意指令 → NONE。小模型學不會「否定」這條規則，只能靠組合硬記
    if NEGATE:
        # 只取「動作型」句子；BRIGHTER/DIMMER 有很多狀態句（太暗了），加否定詞會不通順
        commands = [t for k, ts in CORE.items()
                    if not k.startswith(("BRIGHTER", "DIMMER")) for t in ts if "不" not in t]
        for _ in range(n_per_intent):
            t = random.choice(commands).replace("{c}", random.choice(COLOR_WORDS[random.choice(COLORS[1:])]))
            t = t.replace("{n}", str(random.choice([20, 50, 80])))
            rows.add((random.choice(NEG) + t + random.choice(SUF), "NONE", "NONE"))
    return [{"text": t, "intent": i, "color": c} for t, i, c in rows]


def load_llm(path):
    """讀 llm_sentences.txt：每句收原句 + 1 個加了前後綴的變化版"""
    rows, intent, color = set(), None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            parts = line[2:].split()
            if parts[0] not in INTENTS:
                continue                               # 說明文字
            intent, color = parts[0], parts[1] if len(parts) > 1 else "NONE"
            assert color in COLORS, line
            continue
        rows.add((line, intent, color))
        rows.add((random.choice(PRE) + line + random.choice(SUF), intent, color))
    return [{"text": t, "intent": i, "color": c} for t, i, c in rows]


if __name__ == "__main__":
    import sys
    rows = make()
    llm_path = Path(__file__).parent / "data" / "llm_sentences.txt"
    if "--templates-only" not in sys.argv and llm_path.exists():
        llm_rows = load_llm(llm_path)
        print(f"加入 LLM 句子 {len(llm_rows)} 筆")
        rows += llm_rows
    random.shuffle(rows)
    out = Path(__file__).parent / "data" / "train.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 印統計，確認每類數量大致平衡
    from collections import Counter
    print(f"共 {len(rows)} 筆 → {out}")
    print("intent:", dict(Counter(r["intent"] for r in rows)))
    print("color: ", dict(Counter(r["color"] for r in rows)))
