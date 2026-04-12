#!/usr/bin/env python3
"""Adjust image contrast randomly"""

import tensorflow as tf


def change_contrast(image):
    """Adjusts the contrast of an image randomly"""
    return tf.image.random_contrast(image, lower=0.5, upper=1.5)
