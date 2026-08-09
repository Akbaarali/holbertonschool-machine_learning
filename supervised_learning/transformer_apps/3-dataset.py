#!/usr/bin/env python3
"""Dataset module for Transformer machine translation."""

import transformers
import tensorflow as tf
from setup import load_pt2en


class Dataset:
    """Loads and prepares a dataset for machine translation."""

    def __init__(self, batch_size, max_len):
        """Initialize and prepare the dataset pipeline."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        self.data_train = self.data_train.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )
        )

        self.data_train = self.data_train.cache()

        self.data_train = self.data_train.shuffle(
            buffer_size=20000
        )

        self.data_train = self.data_train.padded_batch(
            batch_size
        )

        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        self.data_valid = self.data_valid.filter(
            lambda pt, en: tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )
        )

        self.data_valid = self.data_valid.padded_batch(
            batch_size
        )

    def tokenize_dataset(self, data):
        """Create Portuguese and English subword tokenizers."""
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )

        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        def pt_iterator():
            """Yield Portuguese sentences."""
            for pt, _ in data:
                yield pt.numpy().decode('utf-8')

        def en_iterator():
            """Yield English sentences."""
            for _, en in data:
                yield en.numpy().decode('utf-8')

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
        """Encode Portuguese and English sentences into token lists."""
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence,
            add_special_tokens=False
        )

        en_tokens = self.tokenizer_en.encode(
            en_sentence,
            add_special_tokens=False
        )

        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = (
            [pt_vocab_size]
            + pt_tokens
            + [pt_vocab_size + 1]
        )

        en_tokens = (
            [en_vocab_size]
            + en_tokens
            + [en_vocab_size + 1]
        )

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper for encode."""
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
