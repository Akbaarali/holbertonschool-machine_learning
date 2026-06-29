#!/usr/bin/env python3
"""Bag of words embedding module."""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use for the analysis

    Returns:
        embeddings: numpy.ndarray of shape (s, f)
        features: list/array of features used for embeddings
    """
    vectorizer = CountVectorizer(vocabulary=vocab)

    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names_out()

    return embeddings, features
