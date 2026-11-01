"""字元 + 2-gram hash tokenizer。

⚠️ ESP32 的 C++ 版必須跟這裡一模一樣，任何差異都會讓模型看到錯的輸入。
"""
MAX_LEN = 32       # 每句最多幾個 token，不足補 0
BUCKETS = 4096     # hash 桶數（編號 0 保留給 padding）
VOCAB_SIZE = BUCKETS + 1   # embedding 表總排數


def tokenize(text):
    # 英文轉小寫，「LED」和「led」視為同一個字
    cps = [ord(c) for c in text.lower()]                   # 字 → Unicode 碼點
    ids = [c % BUCKETS + 1 for c in cps]                   # 單字
    ids += [(a * 31 + b) % BUCKETS + 1                     # 2-gram：相鄰兩字
            for a, b in zip(cps, cps[1:])]
    ids = ids[:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))
