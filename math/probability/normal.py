#!/usr/bin/env python3
"""
This module defines the normal distribution class.
"""


class Normal:
    """Represents an normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes a normal distribution.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Docstring for z_score      
        :param self: Description
        :param x: Description
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Docstring for x_score
        :param self: Description
        :param z: Description
        """
        return z * self.stddev + self.mean
