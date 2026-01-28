#!/usr/bin/env python3
"""Cofactor matrix calculation."""


def _determinant(matrix):
    """Calculates the determinant of a matrix."""
    if matrix == [[]]:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        sub = [row[:col] + row[col + 1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * _determinant(sub)
    return det


def minor(matrix):
    """Calculates the minor matrix of a matrix."""
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minors = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            sub = [r[:j] + r[j + 1:] for k, r in enumerate(matrix) if k != i]
            row_minors.append(_determinant(sub))
        minors.append(row_minors)

    return minors


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix."""
    minors = minor(matrix)
    n = len(minors)

    cofactors = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(((-1) ** (i + j)) * minors[i][j])
        cofactors.append(row)

    return cofactors
