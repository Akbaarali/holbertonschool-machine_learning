#!/usr/bin/env python3
"""Dataset module for Transformer applications."""

import transformers
from setup import load_pt2en


class Dataset:
    """Load and prepare a Portuguese-English translation dataset."""

    def __init__(self):
        """Initialize the Dataset."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English tokenizers from training data."""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def pt_iterator():
            """Yield Portuguese sentences."""
            for pt, _ in data:
                yield pt.numpy().decode("utf-8")

        def en_iterator():
            """Yield English sentences."""
            for _, en in data:
                yield en.numpy().decode("utf-8")

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(),
            vocab_size=2 ** 13
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Encode Portuguese and English sentences into tokens."""
        pt_sentence = pt.numpy().decode("utf-8")
        en_sentence = en.numpy().decode("utf-8")

        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence,
            add_special_tokens=False
        )

        en_tokens = self.tokenizer_en.encode(
            en_sentence,
            add_special_tokens=False
        )

        pt_start = self.tokenizer_pt.vocab_size
        pt_end = self.tokenizer_pt.vocab_size + 1

        en_start = self.tokenizer_en.vocab_size
        en_end = self.tokenizer_en.vocab_size + 1

        pt_tokens = [pt_start] + pt_tokens + [pt_end]
        en_tokens = [en_start] + en_tokens + [en_end]

        return pt_tokens, en_tokens
