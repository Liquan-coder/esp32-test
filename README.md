# 迷你 Jev：在 ESP32-S3 上跑的中文語音燈控小模型

打一句中文（例如「幫我把燈調成藍色」「亮度調到百分之四十五」「別讓 LED 閃」），
一個**在 ESP32 上本地推理**的小型神經網路判斷你的意思，直接控制板載 WS2812 RGB LED。
不用網路、不用雲端 API。

```
「幫我開紅燈」 ──▶ tokenizer ──▶ Embedding 平均 ──▶ 3 層 Dense (INT8) ──▶ intent=ON, color=RED ──▶ 💡
```

## 模型會判斷什麼

| 輸出 | 可能的值 |
|---|---|
| **intent**（要做什麼） | `NONE` `ON` `OFF` `BRIGHTER` `DIMMER` `SET_BRIGHTNESS` `SET_COLOR` `BLINK` |
| **color**（什麼顏色） | `NONE` `RED` `GREEN` `BLUE` `YELLOW` `PURPLE` `WHITE` |

- 否定句（「不要開燈」）會判成 `NONE`
- 亮度數字不靠模型，由 regex 抓（支援 `80`、`四十五`、`百分之三十`），限制在 0~100
- 信心低於門檻 **0.65** 就不動作

## 架構重點

1. **Tokenizer**：單字 + 相鄰 2-gram，hash 到 4096 個桶，每句最多 32 個 token。
   Python（`train/tokenizer.py`）與 C++（`firmware/src/tokenizer.h`）必須逐位元一致。
2. **模型拆成兩半**
   - 前半（查表 + 平均）：在 C++ 用 float 算，embedding 表存成 int8 + 每列一個 scale
   - 後半（3 層 Dense）：TFLite Micro INT8，只用 `FULLY_CONNECTED`
   - 為什麼拆：int8 的 SUM 很容易溢位，電腦 / TFLite 參考實作 / ESP32 會算出三種答案；
     平均交給 float，剩下的矩陣乘法每家實作都一致。
3. **三層驗證**：Python float 模型 → Python INT8 模型 → C++ 版（`tools/check_cpp.py` 逐句比對）

## 專案結構

```
train/                 訓練（Python / TensorFlow）
  schema.py            intent / color 的唯一定義來源
  gen_data.py          模板 + 隨機填空產生訓練資料 → data/train.jsonl
  tokenizer.py         字元 + 2-gram hash tokenizer
  parse.py             抓亮度數字
  model.py / embed.py  模型定義、前半的 embedding 平均
  train.ipynb          訓練、畫學習曲線、混淆矩陣、存 jev_model.keras
  export.py            拆模型 → 量化 → 驗證 → 產生 build/jev_model.h
  data/                train / test / generalize（沒看過的關鍵字）資料集
  build/               匯出的 tflite 與 C 標頭檔（韌體直接 include）
firmware/              ESP32-S3 韌體（PlatformIO + Arduino）
  src/main.cpp         收序列埠一行字 → 推理 → 控制 LED
  src/jev.h            TFLite Micro 推理
  src/led.h            LED 狀態機（亮度、顏色、閃爍）
  test_host/cli.cpp    在 Mac 上跑 C++ tokenizer，給 check_cpp.py 比對
tools/
  jev_tui.py           互動介面：電腦模擬或接 ESP32 看結果
  serial_chat.py       在電腦打字、整句送給 ESP32
  check_cpp.py         驗證 C++ 與 Python 的 tokenizer / 數字解析一致
```

## 硬體

- **ESP32-S3-DevKitC-1 N16R8**（16MB Flash + 8MB Octal PSRAM）
- 板載 WS2812 RGB LED 接 **GPIO48**（其他板子請改 `firmware/src/led.h` 的 `LED_PIN`）

## 快速開始

### 0. 安裝環境

```bash
uv venv --python 3.12
uv pip install -r train/requirements.txt
```

韌體需要 [PlatformIO](https://platformio.org/)（`pio` 指令）。

### 1. 直接燒錄（不用重新訓練）

repo 已附上訓練好的 `train/build/jev_model.h`：

```bash
cd firmware
pio run -t upload        # 編譯 + 燒錄
pio device monitor       # 看輸出，打一句話按 Enter
```

或用電腦端介面跟板子互動：

```bash
uv run python tools/serial_chat.py        # 純文字聊天
uv run python tools/jev_tui.py --serial   # 圖形化顯示模型判斷 + LED 狀態
```

### 2. 只在電腦上玩（不用板子）

```bash
uv run python tools/jev_tui.py            # 跑 INT8 模型
uv run python tools/jev_tui.py --float    # 跑 float32 模型
uv run python tools/jev_tui.py -t 0.8     # 調整信心門檻
```

### 3. 重新訓練

```bash
uv run python train/gen_data.py           # 產生訓練資料
# 用 VS Code / Cursor 開 train/train.ipynb，kernel 選 .venv，一格一格執行 → 存出 jev_model.keras
uv run python train/export.py             # 量化 + 驗證 + 產生 build/jev_model.h
uv run python tools/check_cpp.py          # 確認 C++ 版沒有跑掉
cd firmware && pio run -t upload          # 重新燒錄
```

> 改了 `train/schema.py`、`tokenizer.py` 或 `parse.py`，記得同步改 `firmware/src/` 對應的 `.h`，
> 再跑一次 `check_cpp.py`。

## 編輯器（clangd）

`firmware/.clangd` 和 `.vscode/settings.json` 讓 clangd 能看懂 ESP32 的標頭檔。
`compile_commands.json` 沒放進 repo（含本機路徑），clone 後重新產生：

```bash
cd firmware && pio run -t compiledb
```

`.clangd` 與 `.vscode/settings.json` 裡的 toolchain 路徑是 `~/.platformio/packages/...`，
如果你的使用者名稱不同，要改成自己的路徑。
