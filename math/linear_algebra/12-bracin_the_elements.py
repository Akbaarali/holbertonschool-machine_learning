#!/usr/bin/env python3
"""Module that defines element-wise operations with NumPy arrays."""


def np_elementwise(mat1, mat2):
    """Return element-wise add, sub, mul, and div of mat1 and mat2."""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return add, sub, mul, div
