#!/usr/bin/env python3
"""Dataset module for Portuguese to English machine translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Loads and prepares a Portuguese to English translation dataset."""

    def __init__(self):
        """Initialize the Dataset instance and tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create and train Portuguese and English sub-word tokenizers."""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def portuguese_iterator():
            """Yield Portuguese sentences from the dataset."""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def english_iterator():
            """Yield English sentences from the dataset."""
            for _, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            portuguese_iterator(),
            vocab_size=2 ** 13
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            english_iterator(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
