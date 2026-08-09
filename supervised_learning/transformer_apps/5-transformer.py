#!/usr/bin/env python3
"""Transformer network for machine translation."""

import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """Calculate positional encodings for a Transformer."""
    positions = tf.cast(
        tf.range(max_seq_len)[:, tf.newaxis],
        tf.float32
    )

    dimensions = tf.cast(
        tf.range(dm)[tf.newaxis, :],
        tf.float32
    )

    angle_rates = 1 / tf.pow(
        10000.0,
        (2 * tf.floor(dimensions / 2)) / tf.cast(dm, tf.float32)
    )

    angle_rads = positions * angle_rates

    even_mask = tf.cast(
        tf.range(dm) % 2 == 0,
        tf.float32
    )[tf.newaxis, :]

    odd_mask = 1.0 - even_mask

    return (
        tf.sin(angle_rads) * even_mask
        +
        tf.cos(angle_rads) * odd_mask
    )


def sdp_attention(Q, K, V, mask=None):
    """Calculate scaled dot product attention."""
    matmul_qk = tf.matmul(
        Q,
        K,
        transpose_b=True
    )

    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    scaled_logits = (
        matmul_qk /
        tf.math.sqrt(dk)
    )

    if mask is not None:
        scaled_logits += mask * -1e9

    weights = tf.nn.softmax(
        scaled_logits,
        axis=-1
    )

    output = tf.matmul(
        weights,
        V
    )

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer."""

    def __init__(self, dm, h):
        """Initialize multi-head attention."""
        super(MultiHeadAttention, self).__init__()

        self.dm = dm
        self.h = h
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split a tensor into attention heads."""
        x = tf.reshape(
            x,
            (
                batch_size,
                -1,
                self.h,
                self.depth
            )
        )

        return tf.transpose(
            x,
            perm=[0, 2, 1, 3]
        )

    def call(self, Q, K, V, mask=None):
        """Perform multi-head attention."""
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scaled_attention, weights = sdp_attention(
            Q,
            K,
            V,
            mask
        )

        scaled_attention = tf.transpose(
            scaled_attention,
            perm=[0, 2, 1, 3]
        )

        concat_attention = tf.reshape(
            scaled_attention,
            (
                batch_size,
                -1,
                self.dm
            )
        )

        output = self.linear(
            concat_attention
        )

        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize an encoder block."""
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(
            dm,
            h
        )

        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation="relu"
        )

        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(
            drop_rate
        )

        self.dropout2 = tf.keras.layers.Dropout(
            drop_rate
        )

    def call(self, x, training=False, mask=None):
        """Perform the forward pass."""
        attention, _ = self.mha(
            x,
            x,
            x,
            mask
        )

        attention = self.dropout1(
            attention,
            training=training
        )

        out1 = self.layernorm1(
            x + attention
        )

        ffn = self.dense_hidden(out1)
        ffn = self.dense_output(ffn)

        ffn = self.dropout2(
            ffn,
            training=training
        )

        output = self.layernorm2(
            out1 + ffn
        )

        return output


class DecoderBlock(tf.keras.layers.Layer):
    """Transformer decoder block."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize a decoder block."""
        super(DecoderBlock, self).__init__()

        self.mha1 = MultiHeadAttention(
            dm,
            h
        )

        self.mha2 = MultiHeadAttention(
            dm,
            h
        )

        self.dense_hidden = tf.keras.layers.Dense(
            hidden,
            activation="relu"
        )

        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )

        self.dropout1 = tf.keras.layers.Dropout(
            drop_rate
        )

        self.dropout2 = tf.keras.layers.Dropout(
            drop_rate
        )

        self.dropout3 = tf.keras.layers.Dropout(
            drop_rate
        )

    def call(self, x, encoder_output, training=False,
             look_ahead_mask=None, padding_mask=None):
        """Perform the forward pass."""
        attn1, _ = self.mha1(
            x,
            x,
            x,
            look_ahead_mask
        )

        attn1 = self.dropout1(
            attn1,
            training=training
        )

        out1 = self.layernorm1(
            x + attn1
        )

        attn2, _ = self.mha2(
            out1,
            encoder_output,
            encoder_output,
            padding_mask
        )

        attn2 = self.dropout2(
            attn2,
            training=training
        )

        out2 = self.layernorm2(
            out1 + attn2
        )

        ffn = self.dense_hidden(out2)
        ffn = self.dense_output(ffn)

        ffn = self.dropout3(
            ffn,
            training=training
        )

        output = self.layernorm3(
            out2 + ffn
        )

        return output


class Encoder(tf.keras.layers.Layer):
    """Transformer encoder."""

    def __init__(self, N, dm, h, hidden,
                 input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialize the Transformer encoder."""
        super(Encoder, self).__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(
            input_vocab,
            dm
        )

        self.positional_encoding = positional_encoding(
            max_seq_len,
            dm
        )

        self.blocks = [
            EncoderBlock(
                dm,
                h,
                hidden,
                drop_rate
            )
            for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(
            drop_rate
        )

    def call(self, x, training=False, mask=None):
        """Perform the encoder forward pass."""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)

        x *= tf.math.sqrt(
            tf.cast(self.dm, tf.float32)
        )

        x += self.positional_encoding[
            :seq_len,
            :
        ]

        x = self.dropout(
            x,
            training=training
        )

        for block in self.blocks:
            x = block(
                x,
                training=training,
                mask=mask
            )

        return x


class Decoder(tf.keras.layers.Layer):
    """Transformer decoder."""

    def __init__(self, N, dm, h, hidden,
                 target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialize the Transformer decoder."""
        super(Decoder, self).__init__()

        self.N = N
        self.dm = dm

        self.embedding = tf.keras.layers.Embedding(
            target_vocab,
            dm
        )

        self.positional_encoding = positional_encoding(
            max_seq_len,
            dm
        )

        self.blocks = [
            DecoderBlock(
                dm,
                h,
                hidden,
                drop_rate
            )
            for _ in range(N)
        ]

        self.dropout = tf.keras.layers.Dropout(
            drop_rate
        )

    def call(self, x, encoder_output,
             training=False,
             look_ahead_mask=None,
             padding_mask=None):
        """Perform the decoder forward pass."""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)

        x *= tf.math.sqrt(
            tf.cast(self.dm, tf.float32)
        )

        x += self.positional_encoding[
            :seq_len,
            :
        ]

        x = self.dropout(
            x,
            training=training
        )

        for block in self.blocks:
            x = block(
                x,
                encoder_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask
            )

        return x


class Transformer(tf.keras.Model):
    """Complete Transformer network."""

    def __init__(self, N, dm, h, hidden,
                 input_vocab, target_vocab,
                 max_seq_input, max_seq_target,
                 drop_rate=0.1):
        """Initialize the Transformer."""
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate
        )

        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate
        )

        self.linear = tf.keras.layers.Dense(
            target_vocab
        )

    def call(self, inputs, target,
             training=False,
             encoder_mask=None,
             look_ahead_mask=None,
             decoder_mask=None):
        """Perform the Transformer forward pass."""
        encoder_output = self.encoder(
            inputs,
            training=training,
            mask=encoder_mask
        )

        decoder_output = self.decoder(
            target,
            encoder_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=decoder_mask
        )

        output = self.linear(
            decoder_output
        )

        return output
