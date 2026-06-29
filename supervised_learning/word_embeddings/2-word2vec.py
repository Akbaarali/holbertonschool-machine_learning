#!/usr/bin/env python3
"""Converts a gensim Word2Vec model to a Keras Embedding layer."""

from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a Keras Embedding layer.

    Args:
        model: trained gensim Word2Vec model

    Returns:
        A trainable Keras Embedding layer
    """
    weights = model.wv.vectors

    embedding = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return embedding
