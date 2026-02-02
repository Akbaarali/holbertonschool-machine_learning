#!/usr/bin/env python3
"""
This module defines the Exponential distribution class.
"""


class Exponential:
    """Represents an exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes an Exponential distribution.

        Args:
            data (list): List of data used to estimate lambtha.
            lambtha (float): Expected number of occurrences in a given time frame.

        Raises:
            TypeError: If data is not a list.
            ValueError: If lambtha is not positive or data has fewer than 2 values.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period.

        Args:
            x (float): The time period.

        Returns:
            float: The PDF value for x.
        """
        if x < 0:
            return 0

        e_term = 2.7182818285 ** (-self.lambtha * x)
        return self.lambtha * e_term
