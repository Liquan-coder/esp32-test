// 從句子裡抓亮度數字：C++ 版，必須跟 train/parse.py 的 extract_number 結果一樣
// 只用標準 C++，ESP32 和 Mac 都能編譯（tools/check_cpp.py 會批次比對）
#pragma once
#include <stdint.h>
#include "tokenizer.h"

// 中文數字字元 → 值，不是數字回傳 -1
inline int zh_digit(uint32_t c) {
  switch (c) {
    case 0x96F6: return 0;  // 零
    case 0x4E00: return 1;  // 一
    case 0x4E8C: return 2;  // 二
    case 0x5169: return 2;  // 兩
    case 0x4E09: return 3;  // 三
    case 0x56DB: return 4;  // 四
    case 0x4E94: return 5;  // 五
    case 0x516D: return 6;  // 六
    case 0x4E03: return 7;  // 七
    case 0x516B: return 8;  // 八
    case 0x4E5D: return 9;  // 九
    default: return -1;
  }
}
const uint32_t ZH_TEN = 0x5341;       // 十
const uint32_t ZH_HUNDRED = 0x767E;   // 百

// 對應 Python 的 DIGITS.get(片段, 預設)：片段剛好是 1 個數字字才查得到，否則回傳預設
inline int digits_get(const uint32_t* s, int n, int dflt) {
  if (n == 1 && zh_digit(s[0]) >= 0) return zh_digit(s[0]);
  return dflt;
}

// 中文數字 → 整數，支援 0~100。例：四十五 → 45、十 → 10、一百 → 100。看不懂回傳 -1
inline int chinese_to_int(const uint32_t* s, int n) {
  if (s[n - 1] == ZH_HUNDRED)
    return digits_get(s, n - 1, 1) * 100;
  for (int k = 0; k < n; k++) {
    if (s[k] == ZH_TEN)                // 「十」前面沒字 = 一十；後面沒字 = 整十
      return digits_get(s, k, 1) * 10 + digits_get(s + k + 1, n - k - 1, 0);
  }
  return digits_get(s, n, -1);
}

// 阿拉伯數字：半形 0-9 或全形０-９（Python 的 \d 兩種都認得）
inline int digit_value(uint32_t c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 0xFF10 && c <= 0xFF19) return c - 0xFF10;
  return -1;
}

inline bool is_zh_number(uint32_t c) {
  return zh_digit(c) >= 0 || c == ZH_TEN || c == ZH_HUNDRED;
}

// 回傳句子裡的第一個數字（限制在 0~100），找不到回傳 -1
inline int extract_number(const char* text, int len) {
  uint32_t raw[128], cps[128];
  int n_raw = decode_utf8(text, len, raw, 128);

  // 先拿掉「百分之」，不然「百分之四十五」的「百」會被當成 100
  int n = 0;
  for (int i = 0; i < n_raw; i++) {
    if (i + 2 < n_raw && raw[i] == ZH_HUNDRED && raw[i + 1] == 0x5206 && raw[i + 2] == 0x4E4B) {  // 百 分 之
      i += 2;
      continue;
    }
    cps[n++] = raw[i];
  }

  // 阿拉伯數字優先：找第一段連續數字
  for (int i = 0; i < n; i++) {
    if (digit_value(cps[i]) < 0) continue;
    long v = 0;
    for (; i < n && digit_value(cps[i]) >= 0; i++)
      if (v <= 1000) v = v * 10 + digit_value(cps[i]);   // 超過 1000 就不再累加，反正最後會被限制成 100，也避免溢位
    return v > 100 ? 100 : (int)v;
  }

  // 沒有阿拉伯數字，找第一段連續中文數字
  for (int i = 0; i < n; i++) {
    if (!is_zh_number(cps[i])) continue;
    int start = i;
    while (i < n && is_zh_number(cps[i])) i++;
    int v = chinese_to_int(cps + start, i - start);
    if (v < 0) return -1;
    return v > 100 ? 100 : v;
  }
  return -1;
}
