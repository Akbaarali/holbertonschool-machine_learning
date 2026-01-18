#!/usr/bin/env python3
"""
Task 3: Rename
Renames Timestamp to Datetime, converts it to datetime, and keeps Datetime/Close.
"""

import pandas as pd


def rename(df):
    """Modify df by renaming Timestamp to Datetime, converting it, and selecting columns."""
    df = df.rename(columns={"Timestamp": "Datetime"})

    if pd.api.types.is_numeric_dtype(df["Datetime"]):
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    else:
        df["Datetime"] = pd.to_datetime(df["Datetime"])

    return df[["Datetime", "Close"]]
