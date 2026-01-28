#!/usr/bin/env python3
"""Module that defines cat_matrices2D."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenate two 2D matrices along a given axis."""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        new = []
        for row in mat1:
            new.append(row[:])
        for row in mat2:
            new.append(row[:])
        return new

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        new = []
        for r1, r2 in zip(mat1, mat2):
            new.append(r1[:] + r2[:])
        return new

    return None
