#!/usr/bin/env python3
"""Defines a function that computes the integral of a polynomial."""


def poly_integral(poly, c=0):
    """Return the integral of a polynomial.

    poly is a list where index represents the power of x.
    c is the integration constant.

    If poly or c is invalid, return None.
    The returned list should be as small as possible.
    """
    if type(poly) is not list or len(poly) == 0:
        return None
    if type(c) is not int:
        return None

    for coeff in poly:
        if type(coeff) not in (int, float):
            return None

    integ = [c]
    for i, coeff in enumerate(poly):
        new_coeff = coeff / (i + 1)

        # If the result is a whole number, store it as int
        if type(new_coeff) is float and new_coeff.is_integer():
            new_coeff = int(new_coeff)

        integ.append(new_coeff)

    # Make the list as small as possible (remove trailing zeros)
    while len(integ) > 1 and integ[-1] == 0:
        integ.pop()

    return integ
