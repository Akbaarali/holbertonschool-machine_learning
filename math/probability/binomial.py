#!/usr/bin/env python3
"""
Binomial distribution
"""


class Binomial:
    """Class representing a Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes a Binomial distribution.

        Args:
            data (list): list of data to estimate n and p
            n (int): number of Bernoulli trials
            p (float): probability of success
        """
        if data is None:
            # Use given n and p
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

        else:
            # Estimate n and p from data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # sample mean
            mean = sum(data) / len(data)
            # sample variance
            var = sum((x - mean) ** 2 for x in data) / len(data)
            p_est = 1.0 - (var / mean)
            n_est = round(mean / p_est)
            p_est = mean / n_est

            self.n = int(n_est)
            self.p = float(p_est)
