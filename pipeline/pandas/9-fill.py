#!/usr/bin/env python3
"""
Task 9: Fill
Fill missing values according to specified rules.
"""


def fill(df):
    """Fill missing values and remove Weighted_Price column."""
    df = df.drop(columns=["Weighted_Price"])

    df["Close"] = df["Close"].fillna(method="ffill")

    df["Open"] = df["Open"].fillna(df["Close"])
    df["High"] = df["High"].fillna(df["Close"])
    df["Low"] = df["Low"].fillna(df["Close"])

    df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
    df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

    return df
