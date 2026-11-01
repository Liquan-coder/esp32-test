// tokenizer：C++ 版，必須跟 train/tokenizer.py 算出一模一樣的 token id
// 只用標準 C++，不用 Arduino 的 String，這樣 ESP32 和 Mac 都能編譯（Mac 上跑測試）
#pragma once
#include <stdint.h>

// 把 UTF-8 bytes 拆回 Unicode 碼點，回傳拆出幾個字
inline int decode_utf8(const char* s, int len, uint32_t* cps, int max_n) {
  int n = 0;
  for (int i = 0; i < len && n < max_n; ) {
    uint8_t b = s[i];  // 一個 byte
    if (b < 0x80) {                       // 0xxxxxxx：1 byte（英文、數字）
      cps[n++] = b;
      i += 1;
    } else if ((b & 0xE0) == 0xC0 && i + 1 < len) {   // 110xxxxx：2 byte（é、希臘字母）
      cps[n++] = ((uint32_t)(b & 0x1F) << 6) | (s[i + 1] & 0x3F);
      i += 2;
    } else if ((b & 0xF0) == 0xE0 && i + 2 < len) {   // 1110xxxx：3 byte（中文）；i+2<len 防止句尾只剩半個字時讀出界
      b = s[i] & 0x0f;                    // 0000 1111 utf8編碼第一個byte的前4bits是用來標記的（不放資料）
      uint32_t b0 = b;                    // 再開一個 32bits 的數放b因為左移12位會超過 uint8
      uint32_t b1 = s[++i] & 0x3f;        // 0011 1111 後面的byte的前2bits是用來標記的（不放資料）
      uint32_t b2 = s[++i] & 0x3f;

      b0 = b0 << 12;                      // 剩下 4 6 6 bits 所以最第一個byte要往前(左移，右邊補零) 12位
      b1 = b1 << 6;                       // 同理
      cps[n++] = b0 | b1 | b2;            // 要先把要合起來的數平移，不然全都擠在右邊
      i++;
    } else if ((b & 0xF8) == 0xF0 && i + 3 < len) {   // 11110xxx：4 byte（emoji）
      cps[n++] = ((uint32_t)(b & 0x07) << 18) | ((uint32_t)(s[i + 1] & 0x3F) << 12)
               | ((uint32_t)(s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
      i += 4;
    } else {
      i += 1;                             // 壞掉的 byte（例如半個字）跳過
    }
  }
  return n;
}

const int MAX_LEN = 32;     // 跟 tokenizer.py 一致
const int BUCKETS = 4096;

// 對應 Python 的 str.lower()：英文 A-Z 和全形Ａ-Ｚ轉小寫（訓練資料只出現過半形英文）
inline uint32_t to_lower(uint32_t c) {
  if (c >= 'A' && c <= 'Z') return c + 32;
  if (c >= 0xFF21 && c <= 0xFF3A) return c + 32;   // 全形Ａ(U+FF21) → ａ(U+FF41)
  return c;
}

// 碼點 → token id，寫進 ids（固定 MAX_LEN 格，不足補 0）
inline void tokenize_cps(const uint32_t* cps, int n, int32_t* ids) {
  int k = 0;                                       // 目前寫到 ids 的第幾格
  for (int i = 0; i < n && k < MAX_LEN; i++)       // ① 單字
    ids[k++] = cps[i] % BUCKETS + 1;
  for (int i = 0; i + 1 < n && k < MAX_LEN; i++)   // ② 相鄰兩字：n 個字有 n-1 對
    ids[k++] = (cps[i] * 31 + cps[i + 1]) % BUCKETS + 1;
  while (k < MAX_LEN)                              // ④ 剩下補 0（padding）
    ids[k++] = 0;
}

// 一站完成：UTF-8 字串 → token id。回傳真正的字數（不含 padding）
inline int tokenize(const char* s, int len, int32_t* ids) {
  uint32_t cps[MAX_LEN];                           // 超過 32 個字時，單字 token 就塞滿了，後面的字用不到
  int n = decode_utf8(s, len, cps, MAX_LEN);
  for (int i = 0; i < n; i++) cps[i] = to_lower(cps[i]);
  tokenize_cps(cps, n, ids);
  return n;
}
