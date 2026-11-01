"""第 ③ 層驗證的前半：C++ 版 tokenizer / extract_number 跟 Python 版算出的結果是否完全一樣

用法：uv run python tools/check_cpp.py
會在 Mac 上編譯 firmware/test_host/cli.cpp，餵進所有資料句子 + 邊界案例，逐句比對。
改了 firmware/src/tokenizer.h 或 parse.h 之後跑一次，確認沒改壞。
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from parse import extract_number  # noqa: E402
from tokenizer import tokenize  # noqa: E402

EDGE = [
    # tokenizer：大小寫、全形、emoji、2-byte 字、超過 32 個 token、空白
    "LED開", "Ｌｅｄ開燈", "開燈😀", "café燈", "開" * 40, "亮度 80 %", "a", "燈",
    # extract_number：中文數字、百分之、全形數字、超過 100、多段數字
    "亮度百分之四十五", "調成一百", "亮度十", "亮度十五", "亮度二十", "亮度兩成", "亮度零",
    "百", "亮度９０", "亮度 250", "從 30 調到 70", "三五", "亮度99999999999999", "開燈",
]

texts = EDGE[:]
for f in ["train.jsonl", "test.jsonl", "generalize.jsonl"]:
    texts += [json.loads(l)["text"] for l in open(ROOT / "train/data" / f, encoding="utf-8")]

exe = Path(tempfile.gettempdir()) / "jev_cli"
subprocess.run(["clang++", "-std=c++17", "-O2", str(ROOT / "firmware/test_host/cli.cpp"), "-o", str(exe)], check=True)
out = subprocess.run([str(exe)], input="\n".join(texts) + "\n", capture_output=True, text=True, check=True).stdout

bad = {"tokenizer": 0, "extract_number": 0}
for text, line in zip(texts, out.splitlines()):
    num, ids = line.split("|")
    py_num = extract_number(text)
    for name, cpp, py in [("tokenizer", [int(x) for x in ids.split()], tokenize(text)),
                          ("extract_number", int(num), -1 if py_num is None else py_num)]:
        if cpp != py:
            bad[name] += 1
            if bad[name] <= 5:
                print(f"❌ {name} {text!r}\n   Python: {py}\n   C++   : {cpp}")
for name, n in bad.items():
    print(f"{name:<15} {len(texts)} 句，不一致 {n}")
