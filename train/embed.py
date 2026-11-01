"""模型前半：token id → 查 embedding 表 → 平均（跳過 padding）

⚠️ ESP32 的 firmware/src/jev.h 要跟這裡算法一致：float 逐 token 累加，最後除以真實字數。
"""
import numpy as np


def embed_mean(X, table, scale=None):
    """X: (句數, 32) token id。table 是 float 表，或 int8 表加上每列 scale"""
    X = np.atleast_2d(X)
    vecs = table[X].astype(np.float32)                          # (句數, 32, dim)
    if scale is not None:
        vecs *= scale[X][..., None]                             # int8 → float
    mask = (X > 0)[..., None]
    count = np.maximum(mask.sum(1), 1).astype(np.float32)
    return (vecs * mask).sum(1) / count                         # 全程 float32，跟 C++ 的 float 一樣
