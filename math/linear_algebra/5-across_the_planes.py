#!/usr/bin/env python3
"""Module that defines add_matrices2D."""


def add_matrices2D(mat1, mat2):
    """Add two 2D matrices element-wise and return a new matrix."""
    if len(mat1) != len(mat2):
        return None
    for r1, r2 in zip(mat1, mat2):
        if len(r1) != len(r2):
            return None
    new = []
    for r1, r2 in zip(mat1, mat2):
        new.append([a + b for a, b in zip(r1, r2)])
    return new
