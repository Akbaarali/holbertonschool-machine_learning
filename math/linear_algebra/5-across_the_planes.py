#!/usr/bin/env python3
"""Module that defines add_matrices2D."""


def add_matrices2D(mat1, mat2):
    """Add two 2D matrices element-wise and return a new matrix."""
    if len(mat1) != len(mat2):
        return None
    if any(len(r1) != len(r2) for r1, r2 in zip(mat1, mat2)):
        return None
    return [[r1[i] + r2[i] for i in range(len(r1))] for r1, r2 in zip(mat1, mat2)]
