#!/usr/bin/env python3
"""Dataset preparation for Portuguese-to-English translation."""

from setup import load_pt2en
from transformers import AutoTokenizer


class Dataset:
    """Load and prepare a dataset for machine translation."""

    def __init__(self):
        """Initialize the training data, validation data, and tokenizers."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Create Portuguese and English subword tokenizers.

        Args:
            data: A tf.data.Dataset containing (pt, en) sentence pairs.

        Returns:
            tokenizer_pt: The Portuguese tokenizer.
            tokenizer_en: The English tokenizer.
        """
        pretrained_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            use_fast=True
        )

        pretrained_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            use_fast=True
        )

        def portuguese_sentences():
            """Yield Portuguese sentences as Python strings."""
            for pt, _ in data:
                yield pt.numpy().decode("utf-8")

        def english_sentences():
            """Yield English sentences as Python strings."""
            for _, en in data:
                yield en.numpy().decode("utf-8")

        tokenizer_pt = pretrained_pt.train_new_from_iterator(
            portuguese_sentences(),
            vocab_size=2 ** 13
        )

        tokenizer_en = pretrained_en.train_new_from_iterator(
            english_sentences(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
