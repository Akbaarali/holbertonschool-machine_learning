#!/usr/bin/env python3
"""Adjust image brightness randomly"""

import tensorflow as tf


def change_brightness(image):
    """Adjusts the brightness of an image randomly"""
    return tf.image.random_brightness(image, max_delta=0.2)
