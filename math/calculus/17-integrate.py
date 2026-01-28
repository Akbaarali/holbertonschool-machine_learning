#!/usr/bin/env python3
"""
Task 17 - Integrate
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial.

    poly: list of coefficients where index = power of x
    C: integer integration constant

    Returns:
        New list of coefficients representing the integral, or None if invalid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None

    # poly must contain only ints
    for a in poly:
        if not isinstance(a, int):
            return None

    integ = [C]

    for i, a in enumerate(poly):
        val = a / (i + 1)
        if val.is_integer():
            val = int(val)
        integ.append(val)

    # make list as small as possible (trim trailing zeros)
    while len(integ) > 1 and integ[-1] == 0:
        integ.pop()

    return integ
