#!/usr/bin/env python3
"""Module that defines the integral of a polynomial."""


def poly_integral(poly, c=0):
    """Calculate the integral of a polynomial.

    Args:
        poly (list): list of coefficients where index = power of x
        c (int): integration constant

    Returns:
        list: coefficients of the integrated polynomial
        None: if poly or c is invalid
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(c, int):
        return None

    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    result = [c]

    for i, coef in enumerate(poly):
        value = coef / (i + 1)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        result.append(value)

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
