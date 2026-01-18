#!/usr/bin/env python3
"""
Task 7: Sort
Sort the DataFrame by the High column in descending order.
"""


def high(df):
    """Return DataFrame sorted by High price descending."""
    return df.sort_values(by="High", ascending=False)
