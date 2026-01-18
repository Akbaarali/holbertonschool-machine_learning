#!/usr/bin/env python3
"""
Task 10: Indexing
Set the Timestamp column as the index of the DataFrame.
"""


def index(df):
    """Return df with Timestamp set as the index."""
    return df.set_index("Timestamp")
