#!/usr/bin/env python3
"""Dataset module for Portuguese-English machine translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Load and prepare the Portuguese-English translation dataset."""

    def __init__(self):
        """Initialize the dataset and its tokenizers."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create and train Portuguese and English tokenizers."""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            use_fast=True
        )

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            use_fast=True
        )

        def pt_iterator():
            """Yield batches of Portuguese sentences."""
            batch = []

            for pt, _ in data.as_numpy_iterator():
                batch.append(pt.decode('utf-8'))

                if len(batch) == 1000:
                    yield batch
                    batch = []

            if batch:
                yield batch

        def en_iterator():
            """Yield batches of English sentences."""
            batch = []

            for _, en in data.as_numpy_iterator():
                batch.append(en.decode('utf-8'))

                if len(batch) == 1000:
                    yield batch
                    batch = []

            if batch:
                yield batch

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(),
            vocab_size=2 ** 13
        )

        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
