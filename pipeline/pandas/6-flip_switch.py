#!/usr/bin/env python3
"""
Task 6: Flip it and Switch it
Sort the DataFrame in reverse chronological order and transpose it.
"""


def flip_switch(df):
    """Return df sorted in reverse order, then transposed."""
    return df.sort_index(ascending=False).T
