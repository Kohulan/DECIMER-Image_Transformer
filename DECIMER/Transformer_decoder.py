import numpy as np
import tensorflow as tf

# Set target dtype
TARGET_DTYPE = tf.float32


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding_1d(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=TARGET_DTYPE)


def positional_encoding_2d(row, col, d_model):
    assert d_model % 2 == 0
    row_pos = np.repeat(np.arange(row), col)[:, np.newaxis]
    col_pos = np.repeat(np.expand_dims(np.arange(col), 0), row, axis=0).reshape(-1, 1)

    angle_rads_row = get_angles(
        row_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2
    )
    angle_rads_col = get_angles(
        col_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2
    )

    angle_rads_row[:, 0::2] = np.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = np.cos(angle_rads_row[:, 1::2])
    angle_rads_col[:, 0::2] = np.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = np.cos(angle_rads_col[:, 1::2])
    pos_encoding = np.concatenate([angle_rads_row, angle_rads_col], axis=1)[
        np.newaxis, ...
    ]
    return tf.cast(pos_encoding, dtype=TARGET_DTYPE)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), TARGET_DTYPE)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = tf.cast(mask, TARGET_DTYPE)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], TARGET_DTYPE)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def create_masks_decoder(tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [tf.keras.layers.Dense(dff, activation="relu"), tf.keras.layers.Dense(d_model)]
    )


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, q_pos=None, k_pos=None, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        if q_pos is not None:
            q = q + q_pos
        if k_pos is not None:
            k = k + k_pos

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        scaled_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(scaled_attention)
        return output, attention_weights


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, max_len, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(
        self,
        x,
        enc_output,
        enc_pos,
        dec_pos,
        training,
        look_ahead_mask=None,
        padding_mask=None,
    ):
        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(
            x, x, x, q_pos=dec_pos, k_pos=dec_pos, mask=look_ahead_mask
        )
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            out1, enc_output, enc_output, q_pos=dec_pos, k_pos=enc_pos
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.Model):
    def __init__(
        self, num_layers, d_model, num_heads, dff, target_vocab_size, max_len, rate=0.1
    ):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding_1d = positional_encoding_1d(max_len, d_model)
        self.pos_encoding_2d = positional_encoding_2d(16, 16, self.d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, max_len, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.final_layer = tf.keras.layers.Dense(
            target_vocab_size, activation="softmax"
        )

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        dec_pos = self.pos_encoding_1d[:, :seq_len, :]
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, TARGET_DTYPE))

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x,
                enc_output,
                self.pos_encoding_2d,
                dec_pos,
                training,
                look_ahead_mask,
                padding_mask,
            )

        predictions = self.final_layer(x)
        return predictions
