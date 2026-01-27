#!/usr/bin/env python3
"""Module that provides summation_i_squared."""


def summation_i_squared(n):
    """Return sum of i^2 from i=1 to n. If n is invalid, return None."""
    if type(n) is not int or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
