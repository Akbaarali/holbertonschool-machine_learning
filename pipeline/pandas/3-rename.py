#!/usr/bin/env python3
"""
Task 3: Rename
Renames Timestamp column to Datetime and converts it to datetime.
"""

import pandas as pd


def rename(df):
    """
    Renames Timestamp column to Datetime, converts it to datetime,
    and keeps only Datetime and Close columns.

    Args:
        df (pd.DataFrame): input DataFrame

    Returns:
        pd.DataF
