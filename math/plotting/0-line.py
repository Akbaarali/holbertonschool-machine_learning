#!/usr/bin/env python3
"""
0-line.py

Plots y as a red line graph where y = x^3 for x in [0, 10].
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plot y = x^3 as a solid red line with x-axis ranging from 0 to 10.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)
    plt.plot(x, y, 'r-')
    plt.xlim(0, 10)

    plt.show()
