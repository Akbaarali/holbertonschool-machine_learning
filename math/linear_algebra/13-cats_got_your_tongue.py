#!/usr/bin/env python3
"""Defines a function that concatenates two matrices along a specific axis."""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenate mat1 and mat2 along the given axis."""
    return np.concatenate((mat1, mat2), axis=axis)
