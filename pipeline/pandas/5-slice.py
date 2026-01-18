#!/usr/bin/env python3
"""
Task 5: Slice
Extract specific columns and return every 60th row.
"""


def slice(df):
    """Return every 60th row of selected columns."""
    columns = ["High", "Low", "Close", "Volume_(BTC)"]
    return df[columns].iloc[::60]
