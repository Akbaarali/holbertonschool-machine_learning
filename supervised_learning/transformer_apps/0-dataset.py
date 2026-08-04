#!/usr/bin/env python3
"""Dataset preparation for Portuguese-to-English translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Load and prepare a machine translation dataset."""

    def __init__(self):
        """Initialize the datasets and tokenizers."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English subword tokenizers.

        Args:
            data: A dataset containing Portuguese-English sentence pairs.

        Returns:
            The Portuguese tokenizer and the English tokenizer.
        """
        pretrained_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

        pretrained_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def portuguese_sentences():
            """Yield Portuguese sentences."""
            for pt, _ in data:
                yield pt.numpy().decode("utf-8")

        def english_sentences():
            """Yield English sentences."""
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
