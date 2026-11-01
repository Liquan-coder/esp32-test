"""模型定義。放在獨立檔案，notebook 訓練和 export.py 讀檔都用同一份。"""
import keras
import tensorflow as tf


@keras.saving.register_keras_serializable()   # 讓 .keras 檔讀回來時認得這個自訂層
class MaskedMean(tf.keras.layers.Layer):
    """只平均真的字，跳過補 0 的位置"""
    def call(self, emb, ids):
        mask = tf.cast(ids > 0, emb.dtype)[..., None]              # 真的字=1，padding=0
        return tf.reduce_sum(emb * mask, 1) / tf.maximum(tf.reduce_sum(mask, 1), 1.0)
