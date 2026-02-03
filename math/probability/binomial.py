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
        """CDF at x."""
        e = 2.7182818285

        sqrt2 = 2 ** 0.5
        t = (x - self.mean) / (self.stddev * sqrt2)

        sign = 1
        if t < 0:
            sign = -1
            t = -t

        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        u = 1.0 / (1.0 + p * t)
        poly = (((((a5 * u + a4) * u + a3) * u + a2) * u + a1) * u)
        erf_val = 1.0 - poly * (e ** (-(t * t)))
        erf_val *= sign

        return 0.5 * (1.0 + erf_val)
