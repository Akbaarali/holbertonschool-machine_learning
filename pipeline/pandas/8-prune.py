#!/usr/bin/env python3
"""
Task 8: Prune
Remove rows where the Close column has NaN values.
"""


def prune(df):
    """Return DataFrame with rows having NaN Close values removed."""
    return df.dropna(subset=["Close"])
