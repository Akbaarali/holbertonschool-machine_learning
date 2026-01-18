#!/usr/bin/env python3
"""
Task 0: From Numpy
Creates a pandas.DataFrame from a numpy.ndarray with columns labeled A..Z.
"""

import pandas as pd


def from_numpy(array):
    """
    Creates a pd.DataFrame from a np.ndarray.

    Args:
        array (np.ndarray): numpy array to convert (expected 2D)

    Returns:
        pd.DataFrame: newly created DataFrame with columns A..Z
    """
    if getattr(array, "ndim", None) != 2:
        raise ValueError("array must be a 2D numpy.ndarray")

    n_cols = array.shape[1]
    if n_cols > 26:
        raise ValueError("array must not have more than 26 columns")

    columns = [chr(ord('A') + i) for i in range(n_cols)]
    return pd.DataFrame(array, columns=columns)
