#!/usr/bin/env python3
"""Adjust image contrast randomly"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """Adjusts the contrast of an image randomly"""
    return tf.image.random_contrast(image, lower, upper)
