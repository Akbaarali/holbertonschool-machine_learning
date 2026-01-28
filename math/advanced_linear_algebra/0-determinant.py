#!/usr/bin/env python3
"""Determinant of a matrix."""


def determinant(matrix):
    """Calculates the determinant of a matrix."""
    # Type validation
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if matrix == []:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # 0x0 matrix
    if matrix == [[]]:
        return 1

    # Square validation
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base cases
    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive expansion
    det = 0
    for col in range(n):
        sub = [
            row[:col] + row[col + 1:]
            for row in matrix[1:]
        ]
        det += ((-1) ** col) * matrix[0][col] * determinant(sub)

    return det
