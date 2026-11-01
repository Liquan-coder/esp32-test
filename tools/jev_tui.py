"""迷你 Jev 互動介面：打字 → 看模型怎麼判斷 → 看 LED 怎麼變

用法：
  python tools/jev_tui.py              # 電腦模擬（跑 build/jev_int8.tflite）
  python tools/jev_tui.py --float      # 改跑 float32 版
  python tools/jev_tui.py -t 0.8       # 調整信心門檻
  python tools/jev_tui.py --serial     # 由 ESP32 推理、控制真的 LED（先燒好韌體）
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # 關掉 TensorFlow 的啟動訊息
warnings.filterwarnings("ignore")                  # tf.lite.Interpreter 的「即將改名」提醒
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

import numpy as np                                  # noqa: E402
from rich.console import Console, Group             # noqa: E402
from rich.panel import Panel                        # noqa: E402
from rich.table import Table                        # noqa: E402
from rich.text import Text                          # noqa: E402

from embed import embed_mean                        # noqa: E402
from parse import extract_number                    # noqa: E402
from schema import COLORS, INTENTS                  # noqa: E402
from tokenizer import tokenize                      # noqa: E402

console = Console()

RGB = {"RED": (255, 50, 50), "GREEN": (40, 220, 90), "BLUE": (60, 120, 255),
       "YELLOW": (255, 210, 40), "PURPLE": (170, 80, 255), "WHITE": (255, 255, 255)}
COLOR_ZH = {"RED": "紅", "GREEN": "綠", "BLUE": "藍", "YELLOW": "黃", "PURPLE": "紫", "WHITE": "白"}


# ── 模型 ────────────────────────────────────────────────────────────
class LocalModel:
    """在電腦上跑跟 ESP32 一樣的兩半：前半 float 查表平均（embed.py）→ 後半 TFLite"""

    def __init__(self, path):
        import tensorflow as tf
        self.it = tf.lite.Interpreter(model_path=str(path))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.outs = {d["shape"][-1]: d for d in self.it.get_output_details()}  # 8=intent、7=color
        self.name = path.name
        emb = np.load(path.parent / "jev_embedding.npz")
        # INT8 版用 int8 表（跟 ESP32 一樣）；float 版用原始 float 表
        self.emb = (emb["q"], emb["scale"]) if "int8" in path.name else (emb["table"],)

    def predict(self, tokens):
        t0 = time.perf_counter()
        self.it.set_tensor(self.inp["index"], embed_mean(np.array(tokens), *self.emb))
        self.it.invoke()
        ms = (time.perf_counter() - t0) * 1000
        return self._probs(len(INTENTS)), self._probs(len(COLORS)), ms

    def _probs(self, n):
        d = self.outs[n]
        y = self.it.get_tensor(d["index"])[0].astype(np.float32)
        scale, zero = d["quantization"]
        if scale:
            y = (y - zero) * scale                 # INT8 → float
        e = np.exp(y - y.max())                    # softmax（先減最大值，避免 exp 溢位）
        return e / e.sum()


class SerialModel:
    """把句子送給 ESP32，讀回板子算的結果。推理和 LED 都在板子上，這裡只負責顯示"""

    def __init__(self, port):
        import serial
        self.ser = serial.Serial(port, 115200, timeout=0.1)
        self.name = f"ESP32 @ {port}"
        time.sleep(0.3)
        self.ser.reset_input_buffer()               # 丟掉開機訊息之類的舊輸出

    def predict(self, text):
        self.ser.write((text + "\n").encode("utf-8"))
        deadline = time.time() + 3
        while time.time() < deadline:
            line = self.ser.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("JEV "):              # 只認給程式讀的那行，給人看的跳過
                return json.loads(line[4:])
        raise TimeoutError("3 秒內沒收到板子回應：韌體有燒對嗎？序列埠是不是被別的程式佔用？")


# ── LED 狀態（韌體 firmware/src/led.h 是一樣的邏輯）────────────────────
@dataclass
class Led:
    on: bool = False
    brightness: int = 60
    color: str = "WHITE"
    blink: bool = False

    def apply(self, intent, color, text):
        """回傳一行說明這次做了什麼"""
        self.blink = False
        if color != "NONE" and intent in ("ON", "SET_COLOR", "BLINK"):
            self.color = color
        if intent == "ON":
            self.on = True
            return "開燈"
        if intent == "OFF":
            self.on = False
            return "關燈"
        if intent == "BRIGHTER":
            self.on, self.brightness = True, min(100, self.brightness + 20)  # 燈關著就先打開
            return f"調亮 → {self.brightness}%"
        if intent == "DIMMER":
            self.brightness = max(5, self.brightness - 20)
            return f"調暗 → {self.brightness}%"
        if intent == "SET_BRIGHTNESS":
            n = extract_number(text)
            if n is None:
                return "要設定亮度，但句子裡找不到數字 → 不動作"
            self.on, self.brightness = True, n
            return f"亮度設為 {n}%"
        if intent == "SET_COLOR":
            if color == "NONE":
                return "要換顏色，但沒說哪個顏色 → 不動作"
            self.on = True
            return f"換成{COLOR_ZH[color]}色"
        if intent == "BLINK":
            self.on, self.blink = True, True
            return "閃爍"
        return ""

    def render(self):
        if not self.on:
            return Text("  ●  關", style="grey35")
        r, g, b = (int(c * (0.25 + 0.75 * self.brightness / 100)) for c in RGB[self.color])
        t = Text(" ███ ", style=f"rgb({r},{g},{b})")
        t.append(f" {COLOR_ZH[self.color]}色 {self.brightness}%", style="bold")
        if self.blink:
            t.append("  ✦ 閃爍中", style="bold yellow")
        return t


# ── 畫面 ────────────────────────────────────────────────────────────
def bars(names, probs, top_k=None, highlight=None, width=24):
    table = Table.grid(padding=(0, 1))
    order = np.argsort(probs)[::-1][:top_k]
    for i in order:
        p = float(probs[i])
        style = "bold green" if i == highlight else "white" if p > 0.05 else "grey42"
        table.add_row(Text(f"{names[i]:<15}", style=style),
                      Text(f"{"█" * round(p * width) or "▏":<{width}}", style=style),
                      Text(f"{p:.2f}", style=style))
    return table


def render_turn(text, tokens, p_intent, p_color, ms, threshold, led, action, where="電腦"):
    i, c = int(p_intent.argmax()), int(p_color.argmax())
    intent, color, conf = INTENTS[i], COLORS[c], float(p_intent[i])

    n_real = sum(1 for t in tokens if t)
    tok = Text(f"tokens ({n_real})  ", style="grey50")
    tok.append(" ".join(str(t) for t in tokens[:n_real]), style="grey62")

    if intent == "NONE":
        verdict = Text("➖ NONE：跟燈無關，不動作", style="bold grey70")
    elif conf < threshold:
        verdict = Text(f"⛔ 信心 {conf:.2f} < 門檻 {threshold}，不動作（之後可以轉給雲端）", style="bold red")
    else:
        verdict = Text(f"✅ {intent}" + (f" / {color}" if color != "NONE" else "") + f"  →  {action}",
                       style="bold green")

    grid = Table.grid(expand=True, padding=(0, 3))
    grid.add_column(ratio=3)
    grid.add_column(ratio=2)
    grid.add_row(Panel(bars(INTENTS, p_intent, highlight=i), title="intent", border_style="cyan"),
                 Panel(bars(COLORS, p_color, top_k=4, highlight=c, width=10), title="color", border_style="magenta"))

    stats = Text(f"推理 {ms:.2f} ms（{where}）", style="grey50")
    console.print(Panel(Group(tok, grid, verdict, Text(), Group(Text("LED  ", style="bold") + led.render(), stats)),
                        title=f"[bold]> {text}", title_align="left", border_style="blue"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--float", action="store_true", help="改跑 float32 版模型")
    ap.add_argument("-t", "--threshold", type=float, default=0.65, help="信心門檻（預設 0.65；--serial 時以韌體為準）")
    ap.add_argument("--serial", nargs="?", const="auto", metavar="PORT",
                    help="改由 ESP32 推理、控制真的 LED（不給 PORT 就自動找 /dev/cu.usbmodem*）")
    args = ap.parse_args()
    led = Led()

    if args.serial:
        port = args.serial
        if port == "auto":
            ports = glob.glob("/dev/cu.usbmodem*")
            if not ports:
                console.print("[red]找不到板子：USB 有接上嗎？（ls /dev/cu.* 看看）")
                return
            port = ports[0]
        model = SerialModel(port)
        mode = "實機"
    else:
        path = ROOT / "train/build" / ("jev_float.tflite" if args.float else "jev_int8.tflite")
        if not path.exists():
            console.print(f"[red]找不到 {path}，先執行 python train/export.py")
            return
        with console.status("載入模型中…"):
            model = LocalModel(path)
        mode = "電腦模擬"

    console.print(Panel(f"[bold]迷你 Jev[/]  ·  {model.name}  ·  {mode}  ·  門檻 {args.threshold}\n"
                        "[grey50]輸入指令，例如：開紅燈、亮度調到 30、不要開燈。輸入 q 離開",
                        border_style="blue"))
    while True:
        try:
            text = console.input("[bold blue]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("q", "quit", "exit"):
            break
        if not text:
            continue
        if args.serial:
            try:
                r = model.predict(text)
            except TimeoutError as e:
                console.print(f"[red]{e}")
                continue
            # 板子已經決定好動作、改好 LED，這裡照抄它的狀態來畫
            tokens, action = r["tokens"], r["action"]
            p_intent, p_color, ms = np.array(r["intent"]), np.array(r["color"]), r["us"] / 1000
            led.on, led.brightness, led.blink = bool(r["led"]["on"]), r["led"]["brightness"], bool(r["led"]["blink"])
            led.color = r["led"]["color"]
            render_turn(text, tokens, p_intent, p_color, ms, 0.65, led, action, where="ESP32")
            continue
        tokens = tokenize(text)
        p_intent, p_color, ms = model.predict(tokens)
        i = int(p_intent.argmax())
        action = ""
        if INTENTS[i] != "NONE" and p_intent[i] >= args.threshold:
            action = led.apply(INTENTS[i], COLORS[int(p_color.argmax())], text)
        render_turn(text, tokens, p_intent, p_color, ms, args.threshold, led, action)


if __name__ == "__main__":
    main()
