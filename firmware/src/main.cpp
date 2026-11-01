// 第 4 步：收到一句話 → tokenizer → 模型推理 → 過門檻就控制 LED
#include <Arduino.h>
#include <cstdint>
#include <cstdio>
#include <sys/types.h>
#include "tokenizer.h"
#include "jev.h"
#include "led.h"

String line;   // 還沒收完的這一行
const float THRESHOLD = 0.65f;   // 信心門檻，跟 TUI / export.py 一致
Led led;

void setup() {
  Serial.begin(115200);
  delay(500);
  const char* err = jev::init();
  if (err) {
    Serial.printf("\n模型載入失敗：%s\n", err);
    while (true) delay(1000);          // 停在這裡，不要進 loop
  }
  Serial.printf("\n模型載入成功｜模型 %u bytes｜arena 用了 %u / %d bytes\n",
                jev_model_len, (unsigned)jev::interp->arena_used_bytes(), jev::kArenaSize);
  led.show();                          // 開機先熄燈（燈關著）
  Serial.println("打一句話後按 Enter");
}

void loop() {
  led.tick();                          // 閃爍靠這行：每圈看一次時間，到了就切換亮暗
  while (Serial.available()) {
    char c = Serial.read();                 // 一次只拿到 1 個 byte，不是 1 個字
    if (c != '\r' && c != '\n') {
      line += c;
      continue;
    }
    if (line.length() == 0) continue;       // Enter 可能送 \r\n 兩個 byte，第二個會遇到空行

    Serial.printf("收到「%s」共 %d bytes:", line.c_str(), line.length());
    for (uint i = 0; i < line.length(); i++)
      Serial.printf("%02X ", (uint8_t)line[i]);   // 每個 byte 用兩位十六進位印出
    Serial.println();
    uint32_t cps[32];
    int n = decode_utf8(line.c_str(), line.length(), cps, 32);
    Serial.printf("碼點：");
    for(int i=0;i<n;i++){
     Serial.printf("%u ", (unsigned)cps[i]);
    }
    Serial.println();

    int32_t ids[MAX_LEN];
    tokenize(line.c_str(), line.length(), ids);   // 字串 → 32 個 token id（跟 Python 一樣）
    Serial.printf("token：");
    for (int i = 0; i < MAX_LEN && ids[i]; i++)   // 遇到 0（padding）就停，不印一長串 0
      Serial.printf("%d ", (int)ids[i]);
    Serial.println();

    float p_intent[JEV_N_INTENTS], p_color[JEV_N_COLORS];
    long us = jev::predict(ids, p_intent, p_color);
    int top_i = jev::argmax(p_intent, JEV_N_INTENTS), top_c = jev::argmax(p_color, JEV_N_COLORS);
    Serial.printf("intent：");
    for (int k = 0; k < JEV_N_INTENTS; k++) Serial.printf("%.3f ", p_intent[k]);
    Serial.printf("\ncolor ：");
    for (int k = 0; k < JEV_N_COLORS; k++) Serial.printf("%.3f ", p_color[k]);
    Serial.printf("\n判斷：%s (%.2f) / %s (%.2f)｜%s｜推理 %ld µs\n",
                  JEV_INTENTS[top_i], p_intent[top_i], JEV_COLORS[top_c], p_color[top_c],
                  top_i == 0 ? "NONE，不動作" : p_intent[top_i] >= THRESHOLD ? "✅ 過門檻" : "⛔ 低於門檻",
                  us);

    char msg[96] = "";
    if (top_i != 0 && p_intent[top_i] >= THRESHOLD) {   // NONE 或信心不夠都不動作
      led.apply(JEV_INTENTS[top_i], top_c, line.c_str(), line.length(), msg, sizeof msg);
      Serial.printf("動作：%s\n", msg);
    }
    Serial.printf("LED：%s｜%s色｜亮度 %d%%%s\n", led.on ? "開" : "關", COLOR_ZH[led.color],
                  led.brightness, led.blink ? "｜閃爍中" : "");

    // 給程式讀的一行（tools/jev_tui.py --serial 只認 JEV 開頭這行，上面那些是給人看的）
    Serial.printf("JEV {\"tokens\":[");
    for (int k = 0; k < MAX_LEN; k++) Serial.printf(k ? ",%d" : "%d", (int)ids[k]);
    Serial.printf("],\"intent\":[");
    for (int k = 0; k < JEV_N_INTENTS; k++) Serial.printf(k ? ",%.5f" : "%.5f", p_intent[k]);
    Serial.printf("],\"color\":[");
    for (int k = 0; k < JEV_N_COLORS; k++) Serial.printf(k ? ",%.5f" : "%.5f", p_color[k]);
    Serial.printf("],\"us\":%ld,\"action\":\"%s\",\"led\":{\"on\":%d,\"brightness\":%d,\"color\":\"%s\",\"blink\":%d}}\n\n",
                  us, msg, led.on, led.brightness, JEV_COLORS[led.color], led.blink);
    line = "";
  }
}

