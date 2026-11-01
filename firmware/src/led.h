// LED 狀態機：跟 tools/jev_tui.py 的 Led.apply() 規則一樣，再把狀態真的送到板載 WS2812
#pragma once
#include <Arduino.h>
#include "parse.h"

const int LED_PIN = 48;                // 用紅藍測試找到的：這塊板子的 WS2812 接在 GPIO48
const float LED_MAX = 0.35f;           // 亮度 100% 時只用 35% 的電流，全亮太刺眼
const unsigned long BLINK_MS = 300;    // 閃爍：每 300ms 切換一次

// 顏色順序跟 JEV_COLORS 一樣：NONE, RED, GREEN, BLUE, YELLOW, PURPLE, WHITE
const uint8_t RGB[][3] = {{0, 0, 0}, {255, 50, 50}, {40, 220, 90}, {60, 120, 255},
                          {255, 210, 40}, {170, 80, 255}, {255, 255, 255}};
const char* const COLOR_ZH[] = {"", "紅", "綠", "藍", "黃", "紫", "白"};

struct Led {
  bool on = false;
  int brightness = 60;
  int color = 6;                       // WHITE
  bool blink = false;
  bool blink_phase = true;             // 閃爍時目前是亮還是暗
  unsigned long last_toggle = 0;

  // 依照模型判斷改變狀態，把「做了什麼」寫進 msg。intent / color 是 JEV_INTENTS / JEV_COLORS 的索引
  void apply(const char* intent, int color_idx, const char* text, int len, char* msg, int msg_size) {
    blink = false;
    bool has_color = color_idx != 0;
    if (has_color && (!strcmp(intent, "ON") || !strcmp(intent, "SET_COLOR") || !strcmp(intent, "BLINK")))
      color = color_idx;

    if (!strcmp(intent, "ON")) {
      on = true;
      snprintf(msg, msg_size, "開燈");
    } else if (!strcmp(intent, "OFF")) {
      on = false;
      snprintf(msg, msg_size, "關燈");
    } else if (!strcmp(intent, "BRIGHTER")) {
      on = true;                                     // 燈關著就先打開
      brightness = min(100, brightness + 20);
      snprintf(msg, msg_size, "調亮 → %d%%", brightness);
    } else if (!strcmp(intent, "DIMMER")) {
      brightness = max(5, brightness - 20);
      snprintf(msg, msg_size, "調暗 → %d%%", brightness);
    } else if (!strcmp(intent, "SET_BRIGHTNESS")) {
      int n = extract_number(text, len);
      if (n < 0) {
        snprintf(msg, msg_size, "要設定亮度，但句子裡找不到數字 → 不動作");
      } else {
        on = true;
        brightness = n;
        snprintf(msg, msg_size, "亮度設為 %d%%", n);
      }
    } else if (!strcmp(intent, "SET_COLOR")) {
      if (!has_color) {
        snprintf(msg, msg_size, "要換顏色，但沒說哪個顏色 → 不動作");
      } else {
        on = true;
        snprintf(msg, msg_size, "換成%s色", COLOR_ZH[color_idx]);
      }
    } else if (!strcmp(intent, "BLINK")) {
      on = true;
      blink = true;
      blink_phase = true;
      last_toggle = millis();
      snprintf(msg, msg_size, "閃爍");
    } else {
      msg[0] = '\0';
    }
    show();
  }

  // 每圈 loop() 都呼叫：閃爍用「看時間到了沒」，不用 delay()，才不會卡住收字
  void tick() {
    if (!blink || millis() - last_toggle < BLINK_MS) return;
    last_toggle = millis();
    blink_phase = !blink_phase;
    show();
  }

  // 把目前狀態送到 WS2812
  void show() {
    if (!on || (blink && !blink_phase)) {
      rgbLedWrite(LED_PIN, 0, 0, 0);
      return;
    }
    float k = brightness / 100.0f * LED_MAX;
    rgbLedWrite(LED_PIN, RGB[color][0] * k, RGB[color][1] * k, RGB[color][2] * k);
  }
};
