#!/usr/bin/env python3
"""Defines a function that computes the derivative of a polynomial."""


def poly_derivative(poly):
    """Return the derivative of a polynomial.

    poly is a list where index represents the power of x.
    If poly is invalid, return None.
    If the derivative is 0, return [0].
    """
    if type(poly) is not list or len(poly) == 0:
        return None

    # poly must contain only ints/floats
    for c in poly:
        if type(c) not in (int, float):
            return None

    # derivative: d/dx sum_{i=0}^n poly[i] * x^i = sum_{i=1}^n i*poly[i] * x^(i-1)
    deriv = []
    for i in range(1, len(poly)):
        deriv.append(poly[i] * i)

    # if derivative is all zeros (or poly was constant)
    if len(deriv) == 0:
        return [0]

    all_zero = True
    for v in deriv:
        if v != 0:
            all_zero = False
            break

    return [0] if all_zero else deriv
