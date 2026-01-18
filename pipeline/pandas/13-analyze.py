#!/usr/bin/env python3
"""
Task 13: Analyze
Compute descriptive statistics for all columns except Timestamp.
"""


def analyze(df):
    """
    Return descriptive statistics for df excluding the Timestamp column.
    """
    df = df.drop(columns=["Timestamp"])
    return df.describe()
