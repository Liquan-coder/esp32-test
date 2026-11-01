// 在 ESP32 上跑模型：token id → [前半 float：查表 + 平均] → [後半 TFLite INT8：3 層 Dense] → 機率
// 前半為什麼不交給 TFLite：int8 的 SUM 加總很容易超出範圍，不同實作算出不同答案（見 export.py 開頭）
#pragma once
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <math.h>
#include "jev_model.h"   // export.py 產生：後半模型 + embedding 表 + 類別名稱
#include "tokenizer.h"

namespace jev {

// 模型執行時放中間結果的工作區。太小 AllocateTensors 會失敗，開機時會印出實際用量
constexpr int kArenaSize = 8 * 1024;
alignas(16) inline uint8_t arena[kArenaSize];

inline tflite::MicroInterpreter* interp = nullptr;
inline TfLiteTensor *t_in, *t_intent, *t_color;

// 開機時呼叫一次。失敗回傳原因，成功回傳 nullptr
inline const char* init() {
  const tflite::Model* model = tflite::GetModel(jev_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) return "模型格式版本不符";

  // 只註冊模型用到的運算（export.py 最後一行印的清單）
  static tflite::MicroMutableOpResolver<2> ops;
  ops.AddFullyConnected();
  ops.AddQuantize();

  static tflite::MicroInterpreter it(model, ops, arena, kArenaSize);
  if (it.AllocateTensors() != kTfLiteOk) return "AllocateTensors 失敗（arena 太小或有不支援的運算）";
  interp = &it;

  t_in = it.input(0);
  for (int i = 0; i < 2; i++) {                    // 輸出順序不固定，用形狀辨認：8 類 = intent、7 類 = color
    TfLiteTensor* t = it.output(i);
    (t->dims->data[1] == JEV_N_INTENTS ? t_intent : t_color) = t;
  }
  return nullptr;
}

// 前半：查表 + 平均（跳過 padding）。跟 train/embed.py 的 embed_mean 一樣
inline void embed_mean(const int32_t* ids, float* avg) {
  for (int j = 0; j < JEV_DIM; j++) avg[j] = 0;
  int count = 0;
  for (int t = 0; t < MAX_LEN; t++) {
    int32_t id = ids[t];
    if (id == 0) continue;                         // padding 不算
    count++;
    float s = jev_emb_scale[id];
    for (int j = 0; j < JEV_DIM; j++) avg[j] += jev_emb[id][j] * s;   // int8 × scale → float
  }
  if (count == 0) count = 1;
  for (int j = 0; j < JEV_DIM; j++) avg[j] /= count;
}

// int8 輸出 → 反量化回 float → softmax 變成機率
inline void to_probs(const TfLiteTensor* t, int n, float* p) {
  float mx = -1e9f;
  for (int i = 0; i < n; i++) {
    p[i] = (t->data.int8[i] - t->params.zero_point) * t->params.scale;
    mx = fmaxf(mx, p[i]);
  }
  float sum = 0;
  for (int i = 0; i < n; i++) sum += (p[i] = expf(p[i] - mx));   // 先減最大值，避免 exp 溢位
  for (int i = 0; i < n; i++) p[i] /= sum;
}

// 一句話 → 兩組機率。回傳推理花的微秒數，失敗回傳 -1
inline long predict(const int32_t* ids, float* p_intent, float* p_color) {
  unsigned long t0 = micros();
  embed_mean(ids, t_in->data.f);                   // 前半直接寫進模型的輸入（float，模型自己 QUANTIZE）
  if (interp->Invoke() != kTfLiteOk) return -1;
  long us = micros() - t0;
  to_probs(t_intent, JEV_N_INTENTS, p_intent);
  to_probs(t_color, JEV_N_COLORS, p_color);
  return us;
}

inline int argmax(const float* p, int n) {
  int best = 0;
  for (int i = 1; i < n; i++) if (p[i] > p[best]) best = i;
  return best;
}

}  // namespace jev
