from tensorflow import keras
from tensorflow.keras import layers
from .config import CFG

class TransformerEncoder(layers.Layer):
    def __init__(self, dim, heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(num_heads=heads, key_dim=dim, dropout=dropout)
        self.drop1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(dim),
        ])
        self.drop2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False, mask=None):
        attn = self.attn(x, x, attention_mask=mask, training=training)
        attn = self.drop1(attn, training=training)
        x = self.norm1(x + attn)
        ffn = self.ffn(x, training=training)
        ffn = self.drop2(ffn, training=training)
        x = self.norm2(x + ffn)
        return x


def build_model(cfg=CFG):
    T, H, W = cfg.num_frames, cfg.target_h, cfg.target_w
    inp = layers.Input(shape=(T, H, W, 3), name="video")

    base = keras.applications.MobileNetV2(include_top=False, weights='imagenet', pooling='avg',
                                          input_shape=(H, W, 3))
    base.trainable = False
    x = layers.TimeDistributed(base, name="td_mobilenetv2")(inp)

    # BiLSTM branch
    lstm = layers.Bidirectional(layers.LSTM(cfg.lstm_units, return_sequences=True, dropout=cfg.dropout), name="bilstm")(x)
    lstm_gap = layers.GlobalAveragePooling1D(name="lstm_gap")(lstm)

    # Transformer branch
    t = x
    for i in range(cfg.transformer_layers):
        t = TransformerEncoder(cfg.transformer_dim, cfg.transformer_heads, cfg.transformer_ff_dim, dropout=cfg.dropout, name=f"tx_{i}")(t)
    tx_gap = layers.GlobalAveragePooling1D(name="tx_gap")(t)

    # Fuse + Head
    fused = layers.Concatenate(name="fuse")([lstm_gap, tx_gap])
    fused = layers.Dense(256, activation='gelu')(fused)
    fused = layers.Dropout(cfg.dropout)(fused)
    out = layers.Dense(1, activation='sigmoid', dtype='float32', name="pred")(fused)

    return keras.Model(inputs=inp, outputs=out, name="Hybrid_CNN_BiLSTM_Transformer")
