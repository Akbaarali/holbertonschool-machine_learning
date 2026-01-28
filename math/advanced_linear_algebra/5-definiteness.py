#!/usr/bin/env python3
"""Definiteness of a matrix.

Classifies a matrix as positive/negative (semi-)definite or indefinite.
"""


import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix.

    Args:
        matrix (numpy.ndarray): array of shape (n, n)

    Returns:
        str | None: One of:
            "Positive definite"
            "Positive semi-definite"
            "Negative semi-definite"
            "Negative definite"
            "Indefinite"
        or None if matrix is not a valid matrix for this classification.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2:
        return None

    n, m = matrix.shape
    if n != m or n == 0:
        return None

    # Definiteness is defined (in this context) for real symmetric matrices
    if np.iscomplexobj(matrix):
        return None

    if not np.all(np.isfinite(matrix)):
        return None

    if not np.allclose(matrix, matrix.T, atol=1e-8):
        return None

    # Use eigvalsh for symmetric matrices (real eigenvalues)
    eigvals = np.linalg.eigvalsh(matrix.astype(float))
    tol = 1e-8

    pos = np.any(eigvals > tol)
    neg = np.any(eigvals < -tol)

    if pos and neg:
        return "Indefinite"
    if np.all(eigvals > tol):
        return "Positive definite"
    if np.all(eigvals >= -tol) and np.any(np.abs(eigvals) <= tol):
        return "Positive semi-definite"
    if np.all(eigvals < -tol):
        return "Negative definite"
    if np.all(eigvals <= tol) and np.any(np.abs(eigvals) <= tol):
        return "Negative semi-definite"

    return None
