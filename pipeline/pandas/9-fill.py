#!/usr/bin/env python3
"""
Task 9: Fill
Fill missing values according to specified rules.
"""


def fill(df):
    """Fill missing values and remove Weighted_Price column."""
    df = df.drop(columns=["Weighted_Price"])

    df["Close"] = df["Close"].fillna(method="ffill")

    df[["Open", "High", "Low"]] = df[["Open", "High", "Low"]].fillna(
        df["Close"], axis=0
    )

    df[["Volume_(BTC)", "Volume_(Currency)"]] = df[
        ["Volume_(BTC)", "Volume_(Currency)"]
    ].fillna(0)

    return df
