#!/usr/bin/env python3
"""
Task 4: To Numpy
Return the last 10 rows of the High and Close columns as a numpy.ndarray.
"""


def array(df):
    """Return last 10 rows of High and Close as a numpy array."""
    return df[["High", "Close"]].tail(10).to_numpy()
