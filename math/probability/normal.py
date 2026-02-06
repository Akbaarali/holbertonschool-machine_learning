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
        if x is None:
            return None

        if not isinstance (x, (int, float)):
            return None

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        sign  = 1
        if z < 0:
            sign = -1
            z = -z
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        t = 1.0 / (1.0 + p * z)
        e_const = 2.718281828459045
        exp_term = e_const ** (-(z * z))

        erf_approx = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * exp_term
        erf_approx *= sign
        return 0.5 * (1.0 + erf_approx)
