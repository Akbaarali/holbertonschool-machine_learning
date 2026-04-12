#!/usr/bin/env python3
"""Rotate image 90 degrees counterclockwise"""

import tensorflow as tf


def rotate_image(image):
    """Rotates an image by 90 degrees counterclockwise"""
    return tf.image.rot90(image)
