#!/usr/bin/env python3
"""Create masks for a Transformer network."""

import tensorflow as tf


def create_masks(inputs, target):
    """
    Create all masks needed for Transformer training.

    Args:
        inputs: tensor containing the input sequences
        target: tensor containing the target sequences

    Returns:
        encoder_mask: padding mask for the encoder
        combined_mask: look-ahead and padding mask for the decoder
        decoder_mask: padding mask for the decoder's second attention block
    """
    # Encoder padding mask
    encoder_mask = tf.cast(
        tf.math.equal(inputs, 0),
        tf.float32
    )
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Decoder padding mask for encoder output
    decoder_mask = tf.cast(
        tf.math.equal(inputs, 0),
        tf.float32
    )
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Target padding mask
    target_padding_mask = tf.cast(
        tf.math.equal(target, 0),
        tf.float32
    )
    target_padding_mask = target_padding_mask[
        :, tf.newaxis, tf.newaxis, :
    ]

    # Look-ahead mask
    seq_len = tf.shape(target)[1]

    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len, seq_len)),
        -1,
        0
    )

    # Combine target padding and look-ahead masks
    combined_mask = tf.maximum(
        target_padding_mask,
        look_ahead_mask
    )

    return encoder_mask, combined_mask, decoder_mask
