"""jev_model.keras → 拆成兩半 → 驗證 → C 標頭檔

  前半（查表 + 平均）：用 float 在 C++ 算，embedding 表存成 int8 + 每列一個 scale
  後半（3 層 Dense）：TFLite INT8，只用最穩定的 FULLY_CONNECTED

為什麼拆：int8 的 SUM 把幾十個數加起來很容易超出 int8 範圍，電腦 / TFLite 參考實作 / ESP32
三種實作算出三種答案。平均交給 float，TFLite 只剩每家實作都一致的矩陣乘法。

用法：uv run python train/export.py
輸出：build/jev_float.tflite、build/jev_int8.tflite（後半）、build/jev_embedding.npz（前半）、build/jev_model.h
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import model  # noqa: F401  註冊 MaskedMean，load_model 才認得
from embed import embed_mean
from schema import COLORS, INTENTS
from tokenizer import tokenize

HERE = Path(__file__).parent
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)
THRESHOLD = 0.65   # 跟 UI / 韌體的信心門檻一致


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    X = np.array([tokenize(r["text"]) for r in rows], dtype=np.int32)
    return rows, X


def quantize_table(table):
    """embedding 表 → int8 + 每列一個 scale（對稱量化：0 還是 0，不用 zero_point）"""
    scale = np.abs(table).max(1) / 127
    scale[scale == 0] = 1.0                                     # 全 0 的列（例如沒用到的桶）
    q = np.round(table / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def head_model(keras_model):
    """後半：32 維平均向量 → Dense → intent / color，權重直接搬原模型的"""
    x = tf.keras.Input((keras_model.get_layer("embedding").output_dim,), name="avg")

    def copy_dense(name, inp):
        old = keras_model.get_layer(name)
        new = tf.keras.layers.Dense(old.units, activation=old.activation, name=name)
        y = new(inp)
        new.set_weights(old.get_weights())
        return y

    h = copy_dense("dense", x)
    return tf.keras.Model(x, {"intent": copy_dense("intent", h), "color": copy_dense("color", h)})


def convert(head, int8, A_calib):
    conv = tf.lite.TFLiteConverter.from_keras_model(head)
    if int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        # 量化需要看「真實資料流過時，每層數值的範圍」，才能決定 scale / zero_point
        conv.representative_dataset = lambda: ([a[None]] for a in A_calib)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_output_type = tf.int8      # 輸入維持 float，模型自己 QUANTIZE
        # Dense 權重整層共用一個 scale。新版轉換器預設「每個輸出各一個 scale」（per-channel），
        # 但 ESP32 上這版 TFLite Micro 的 FULLY_CONNECTED 只讀第一個 scale，其他輸出全部算錯
        conv._experimental_disable_per_channel_quantization_for_dense_layers = True
    return conv.convert()


def run(tflite_bytes, A, resolver=tf.lite.experimental.OpResolverType.AUTO):
    """用 TFLite 直譯器逐句推理後半，回傳 (intent logits, color logits, 用到的運算)"""
    it = tf.lite.Interpreter(model_content=tflite_bytes, experimental_op_resolver_type=resolver)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    # 輸出順序不一定跟 Keras 一樣，用形狀辨認：8 類是 intent、7 類是 color
    outs = {d["shape"][-1]: d for d in it.get_output_details()}
    li, lc = [], []
    for a in A:
        it.set_tensor(inp["index"], a[None].astype(np.float32))
        it.invoke()
        for d, store in [(outs[len(INTENTS)], li), (outs[len(COLORS)], lc)]:
            y = it.get_tensor(d["index"])[0].astype(np.float32)
            scale, zero = d["quantization"]
            if scale:                              # INT8 → 反量化回 float
                y = (y - zero) * scale
            store.append(y)
    ops = sorted({op["op_name"] for op in it._get_ops_details()} - {"DELEGATE"})
    return np.array(li), np.array(lc), ops


def c_array(values, fmt, per_line):
    return ",".join(fmt(v) + ("\n  " if i % per_line == per_line - 1 else "") for i, v in enumerate(values))


def write_header(tflite_bytes, emb_q, emb_scale, path):
    path.write_text(
        "// 由 train/export.py 自動產生，不要手動修改\n#pragma once\n#include <stdint.h>\n\n"
        "// 後半：TFLite INT8 模型（3 層 Dense）。alignas(16)：TFLite Micro 要求模型資料對齊\n"
        f"alignas(16) const unsigned char jev_model[] = {{\n  {c_array(tflite_bytes, lambda b: f'0x{b:02x}', 16)}\n}};\n"
        f"const unsigned int jev_model_len = {len(tflite_bytes)};\n\n"
        "// 前半：embedding 表，int8 + 每列一個 scale。第 id 列的真實值 = jev_emb[id][j] * jev_emb_scale[id]\n"
        f"const int JEV_VOCAB = {emb_q.shape[0]};\nconst int JEV_DIM = {emb_q.shape[1]};\n"
        f"const int8_t jev_emb[{emb_q.shape[0]}][{emb_q.shape[1]}] = {{\n  {c_array(emb_q.ravel(), str, 32)}\n}};\n"
        f"const float jev_emb_scale[{len(emb_scale)}] = {{\n  {c_array(emb_scale, lambda s: f'{s:.9g}f', 8)}\n}};\n\n"
        # 類別名稱也一起寫進來，韌體不用手抄 schema.py，改了也不會對不上
        f"const char* const JEV_INTENTS[] = {{{', '.join(f'\"{s}\"' for s in INTENTS)}}};\n"
        f"const char* const JEV_COLORS[] = {{{', '.join(f'\"{s}\"' for s in COLORS)}}};\n"
        f"const int JEV_N_INTENTS = {len(INTENTS)};\n"
        f"const int JEV_N_COLORS = {len(COLORS)};\n")


if __name__ == "__main__":
    keras_model = tf.keras.models.load_model(HERE / "jev_model.keras")
    rows, X_test = load(HERE / "data/test.jsonl")
    _, X_train = load(HERE / "data/train.jsonl")
    _, X_gen = load(HERE / "data/generalize.jsonl")
    yi = np.array([INTENTS.index(r["intent"]) for r in rows])
    yc = np.array([COLORS.index(r["color"]) for r in rows])
    ref = keras_model.predict(X_test, verbose=0)

    # 前半：float 表給 --float 用；int8 表給 ESP32（和預設的 TUI）用
    table = keras_model.get_layer("embedding").get_weights()[0]
    emb_q, emb_scale = quantize_table(table)
    np.savez(BUILD / "jev_embedding.npz", table=table, q=emb_q, scale=emb_scale)
    head = head_model(keras_model)

    print(f"{'版本':<10}{'大小':>10}{'intent':>9}{'color':>8}{'跟 Keras 答案一致':>18}")
    print(f"{'Keras':<10}{'-':>10}{(ref['intent'].argmax(1) == yi).mean():>9.1%}{(ref['color'].argmax(1) == yc).mean():>8.1%}")

    A_calib = embed_mean(X_train[:300], emb_q, emb_scale)
    for name, int8 in [("float32", False), ("int8", True)]:
        b = convert(head, int8, A_calib)
        (BUILD / f"jev_{name.replace('32', '')}.tflite").write_bytes(b)
        A = embed_mean(X_test, emb_q, emb_scale) if int8 else embed_mean(X_test, table)
        li, lc, ops = run(b, A)
        same = ((li.argmax(1) == ref["intent"].argmax(1)) & (lc.argmax(1) == ref["color"].argmax(1))).mean()
        size = len(b) + (emb_q.nbytes + emb_scale.nbytes if int8 else table.nbytes)
        print(f"{name:<10}{size / 1024:>8.1f}KB{(li.argmax(1) == yi).mean():>9.1%}{(lc.argmax(1) == yc).mean():>8.1%}{same:>18.1%}")
        if int8:
            write_header(b, emb_q, emb_scale, BUILD / "jev_model.h")
            int8_bytes = b

    # 答案一樣還不夠：信心值偏移可能讓「過門檻」的判斷改變
    X_all = np.concatenate([X_test, X_gen, X_train[:500]])
    A_all = embed_mean(X_all, emb_q, emb_scale)
    p_ref = tf.nn.softmax(keras_model.predict(X_all, verbose=0)["intent"]).numpy()
    li, _, _ = run(int8_bytes, A_all)
    p_q = tf.nn.softmax(li).numpy()
    conf_ref, conf_q = p_ref.max(1), p_q.max(1)
    diff = np.abs(conf_ref - conf_q)
    flips = ((conf_ref >= THRESHOLD) != (conf_q >= THRESHOLD)).sum()
    print(f"\nINT8 信心值偏移（{len(X_all)} 句）：平均 {diff.mean():.4f}｜最大 {diff.max():.4f}"
          f"｜門檻 {THRESHOLD} 判斷改變 {flips} 句")
    for k in np.argsort(diff)[::-1][:3]:
        print(f"  偏移最大：{conf_ref[k]:.3f} → {conf_q[k]:.3f}")

    # 換一套 TFLite 實作（參考實作，跟 TFLite Micro 同源），結果應該一樣：這次就是被它抓到 SUM 的問題
    li_ref, _, _ = run(int8_bytes, A_all, tf.lite.experimental.OpResolverType.BUILTIN_REF)
    print(f"最佳化實作 vs 參考實作：intent 機率最大差異 "
          f"{np.abs(tf.nn.softmax(li_ref).numpy() - p_q).max():.4f}（應該接近 0）")

    print(f"\n用到的運算（C++ 端要註冊這些）：{', '.join(ops)}")
    print(f"C 標頭檔 → {BUILD / 'jev_model.h'}")
