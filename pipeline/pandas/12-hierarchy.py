#!/usr/bin/env python3
"""
Task 12: Hierarchy
Create a MultiIndex with Timestamp as the first level and exchange as second.
"""

import pandas as pd


def hierarchy(df1, df2):
    """Return concatenated data with Timestamp as first MultiIndex level."""
    index = __import__('10-index').index

    df1 = index(df1)
    df2 = index(df2)

    start = 1417411980
    end = 1417417980

    df1 = df1.loc[(df1.index >= start) & (df1.index <= end)]
    df2 = df2.loc[(df2.index >= start) & (df2.index <= end)]

    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    df.index = df.index.swaplevel(0, 1)
    df = df.sort_index(level=0)

    return df
