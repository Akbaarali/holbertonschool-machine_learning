#!/usr/bin/env python3
"""
Task 4: To Numpy
Convert last 10 rows of High and Close columns to a numpy array.
"""

import pandas as pd


def array(df):
    """
    Select the last 10 rows of High and Close columns
    and return them as a numpy.ndarray.
    """
    return df[["High", "Close"]].tail(10).to_numpy()
