#!/usr/bin/env python3
"""
Normal distribution
"""


class Normal:
    """Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Create a Normal distribution."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            n = len(data)
            mu = sum(data) / n

            # population variance: divide by n (NOT n - 1)
            var = 0
            for x in data:
                var += (x - mu) ** 2
            var = var / n

            self.mean = float(mu)
            self.stddev = float(var ** 0.5)

    def z_score(self, x):
        """z-score of x."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """x-value of z."""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """PDF at x."""
        pi = 3.1415926536
        e = 2.7182818285

        z = (x - self.mean) / self.stddev
        coef = 1 / (self.stddev * ((2 * pi) ** 0.5))
        expo = e ** (-(z ** 2) / 2)
        return coef * expo

    def cdf(self, x):
        """
        Calculates cumilative distribution
        """
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        pi = 3.1415926536
        two_over_sqrt_pi = 2 / (pi ** 0.5)
        term = z
        series_sum = term

        for n in range (0, 49):
            term *= (-1) * (z * z) * (2 * n + 1) / ((n+1) * (2 * n + 3))
            series_sum += term

        erf = two_over_sqrt_pi * series_sum
        return 0.5 * (1 + erf)
