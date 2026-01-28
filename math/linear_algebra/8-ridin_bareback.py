#!/usr/bin/env python3
"""Module that defines mat_mul."""


def mat_mul(mat1, mat2):
    """Multiply two matrices and return a new matrix."""
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for row in mat1:
        new_row = []
        for j in range(len(mat2[0])):
            total = 0
            for i in range(len(mat2)):
                total += row[i] * mat2[i][j]
            new_row.append(total)
        result.append(new_row)

    return result
