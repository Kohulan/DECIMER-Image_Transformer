import tensorflow as tf
import numpy as np

# Set target dtype
TARGET_DTYPE = tf.float32


def get_angles(pos, i, d_model):
    angle_rates = tf.constant(1, TARGET_DTYPE) / tf.math.pow(
        tf.constant(10000, TARGET_DTYPE),
        (tf.constant(2, dtype=TARGET_DTYPE) * tf.cast((i // 2), TARGET_DTYPE))
        / d_model,
    )
    return pos * angle_rates


def do_interleave(arr_a, arr_b):
    a_arr_tf_column = tf.range(arr_a.shape[1]) * 2  # [0 2 4 ...]
    b_arr_tf_column = tf.range(arr_b.shape[1]) * 2 + 1  # [1 3 5 ...]
    column_indices = tf.argsort(tf.concat([a_arr_tf_column, b_arr_tf_column], axis=-1))
    column, row = tf.meshgrid(column_indices, tf.range(arr_a.shape[0]))
    combine_indices = tf.stack([row, column], axis=-1)
    combine_value = tf.concat([arr_a, arr_b], axis=1)
    return tf.gather_nd(combine_value, combine_indices)


def positional_encoding_1d(position, d_model):
    angle_rads = get_angles(
        tf.cast(tf.range(position)[:, tf.newaxis], TARGET_DTYPE),
        tf.cast(tf.range(d_model)[tf.newaxis, :], TARGET_DTYPE),
        d_model,
    )

    # apply sin to even indices in the array; 2i
    sin_angle_rads = tf.math.sin(angle_rads[:, ::2])
    cos_angle_rads = tf.math.cos(angle_rads[:, 1::2])
    angle_rads = do_interleave(sin_angle_rads, cos_angle_rads)
    pos_encoding = angle_rads[tf.newaxis, ...]
    return pos_encoding


def np_positional_encoding_2d(row, col, d_model):
    assert d_model % 2 == 0
    row_pos = np.repeat(np.arange(row), col)[:, np.newaxis]
    col_pos = np.repeat(np.expand_dims(np.arange(col), 0), row, axis=0).reshape(-1, 1)

    angle_rads_row = get_angles(
        row_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2
    ).numpy()
    angle_rads_col = get_angles(
        col_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2
    ).numpy()

    angle_rads_row[:, 0::2] = np.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = np.cos(angle_rads_row[:, 1::2])
    angle_rads_col[:, 0::2] = np.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = np.cos(angle_rads_col[:, 1::2])
    pos_encoding = np.concatenate([angle_rads_row, angle_rads_col], axis=1)[
        np.newaxis, ...
    ]
    return tf.cast(pos_encoding, dtype=TARGET_DTYPE)


def positional_encoding_2d(row, col, d_model):
    row_pos = tf.repeat(tf.range(row), col)[:, tf.newaxis]
    col_pos = tf.reshape(
        tf.repeat(tf.expand_dims(tf.range(col), 0), row, axis=0), (-1, 1)
    )

    angle_rads_row = get_angles(
        tf.cast(row_pos, tf.float32),
        tf.range(d_model // 2)[tf.newaxis, :],
        d_model // 2,
    )
    angle_rads_col = get_angles(
        tf.cast(col_pos, tf.float32),
        tf.range(d_model // 2)[tf.newaxis, :],
        d_model // 2,
    )

    sin_angle_rads_row = tf.math.sin(angle_rads_row[:, ::2])
    cos_angle_rads_row = tf.math.cos(angle_rads_row[:, 1::2])
    angle_rads_row = do_interleave(sin_angle_rads_row, cos_angle_rads_row)

    sin_angle_rads_col = tf.math.sin(angle_rads_col[:, ::2])
    cos_angle_rads_col = tf.math.cos(angle_rads_col[:, 1::2])
    angle_rads_col = do_interleave(sin_angle_rads_col, cos_angle_rads_col)

    pos_encoding = tf.concat([angle_rads_row, angle_rads_col], axis=1)[tf.newaxis, ...]
    return pos_encoding


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), TARGET_DTYPE)

    # add extra dimensions to add the padding to the attention logits.
    #    - (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # (seq_len, seq_len)
    return tf.cast(mask, TARGET_DTYPE)


def create_mask(inp, tar):

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return tf.cast(combined_mask, TARGET_DTYPE)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], TARGET_DTYPE)

    # Calculate scaled attention logits
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k)
    # so that the scores add up to 1.
    #   - shape --> (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    #   - shape --> (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


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

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # (batch_size, seq_len, d_model)
        q = self.wq(q)
        # (batch_size, seq_len, d_model)
        k = self.wk(k)
        # (batch_size, seq_len, d_model)
        v = self.wv(v)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape – (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape – (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            # INNER LAYER
            #   – (batch_size, seq_len, dff)
            tf.keras.layers.Dense(dff, activation="relu"),
            # OUTPUT
            #   – (batch_size, seq_len, d_model)
            tf.keras.layers.Dense(d_model),
        ]
    )


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):

        super(TransformerEncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads,
            key_dim=d_model,
        )
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):

        # returns (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask, return_attention_scores=True)

        # Potentially unncessary by passing dropout1 to tf.keras.layers.MultiHeadAttention (if using tf MHA)
        attn_output = self.dropout1(attn_output, training=training)

        # Residual connection followed by layer normalization
        #   – returns (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output, training=training)

        # Point-wise Feed Forward Step
        #   – returns (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)

        # Residual connection followed by layer normalization
        #   – returns (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output, training=training)

        return out2


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):

        super(TransformerDecoderLayer, self).__init__()

        # WE COULD USE A CUSTOM DEFINED MHA MODEL BUT WE WILL USE TFA INSTEAD
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        #
        # # Multi Head Attention Layers
        # self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=d_model,)
        # self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=d_model,)

        # Feed Forward NN
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Layer Normalization Layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout Layers
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    # enc_output.shape == (batch_size, input_seq_len, d_model)
    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)

        # Residual connection followed by layer normalization
        #   – (batch_size, target_seq_len, d_model)
        out1 = self.layernorm1(attn1 + x, training=training)

        # Merging connection between encoder and decoder (MHA)
        #   – (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)

        # Residual connection followed by layer normalization
        #   – (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1, training=training)

        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)

        # Residual connection followed by layer normalization
        #   – (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2, training=training)

        return out3, attn_weights_block1, attn_weights_block2


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        maximum_position_encoding,
        dropout_rate=0.1,
    ):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Dense(self.d_model, activation="relu")
        self.pos_encoding = positional_encoding_1d(
            maximum_position_encoding, self.d_model
        )
        self.enc_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        # adding embedding and position encoding.
        #   – (batch_size, input_seq_len, d_model)
        x = self.embedding(x)
        x += self.pos_encoding
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        #   – (batch_size, input_seq_len, d_model)
        return x


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
        dropout_rate=0.1,
    ):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding_1d(maximum_position_encoding, d_model)

        self.dec_layers = [
            TransformerDecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask=None, padding_mask=None):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # adding embedding and position encoding.
        #   – (batch_size, target_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, TARGET_DTYPE))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )
            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        pe_input,
        pe_target,
        dropout_rate=0.1,
    ):

        super(Transformer, self).__init__()

        self.t_encoder = TransformerEncoder(
            num_layers, d_model, num_heads, dff, pe_input, dropout_rate
        )
        self.t_decoder = TransformerDecoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            pe_target,
            dropout_rate,
        )
        self.t_final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(
        self,
        t_inp,
        t_tar,
        training,
        enc_padding_mask=None,
        look_ahead_mask=None,
        dec_padding_mask=None,
    ):
        # (batch_size, inp_seq_len, d_model)
        enc_output = self.t_encoder(t_inp, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.t_decoder(
            t_tar, enc_output, training, look_ahead_mask, dec_padding_mask
        )

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.t_final_layer(dec_output)

        return final_output, attention_weights
