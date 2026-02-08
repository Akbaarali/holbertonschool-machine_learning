#!/usr/bin/env python3
"""
0-likelihood.py
"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining the data x and n
    given various hypothetical probabilities P.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients
        P (numpy.ndarray): 1D array of hypothetical probabilities

    Returns:
        numpy.ndarray: likelihood values for each probability in P
    """

    # Error checks
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Binomial coefficient: n! / (x! * (n-x)!)
    factorial = np.math.factorial
    comb = factorial(n) / (factorial(x) * factorial(n - x))

    # Likelihood calculation
    likelihoods = comb * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
