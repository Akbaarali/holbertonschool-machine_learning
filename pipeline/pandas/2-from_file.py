#!/usr/bin/env python3
"""
Task 2: From File
Loads data from a file into a pandas DataFrame.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Loads data from a file as a pandas DataFrame.

    Args:
        filename (str): path to the file
        delimiter (str): column separator

    Returns:
        pd.DataFrame: loaded DataFrame
    """
    return pd.read_csv(filename, sep=delimiter)
