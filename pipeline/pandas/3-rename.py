#!/usr/bin/env python3
"""
Task 3: Rename

Rename Timestamp to Datetime, convert it to datetime, and keep
only Datetime and Close.
"""

import pandas as pd


def rename(df):
    """
    Rename Timestamp to Datetime, convert it to datetime, and keep
    only Datetime and Close columns.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})

    if pd.api.types.is_numeric_dtype(df["Datetime"]):
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    else:
        df["Datetime"] = pd.to_datetime(df["Datetime"])

    return df[["Datetime", "Close"]]
