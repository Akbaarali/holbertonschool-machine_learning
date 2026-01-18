#!/usr/bin/env python3
"""
Task 11: Concat
Concatenate bitstamp and coinbase dataframes.
"""

import pandas as pd


def concat(df1, df2):
    """Concatenate selected df2 rows on top of df1 using Timestamp as index."""
    index = __import__('10-index').index

    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[df2.index <= 1417411920]

    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
