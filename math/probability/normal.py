#!/usr/bin/env python3
"""
Normal distribution
"""

class Normal:
    """Class representing a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes a Normal distribution.

        Args:
            data (list): list of data to estimate mean and stddev
            mean (float): mean of the distribution
            stddev (float): standard deviation of the distribution
        """
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
            var = sum((x - mu) ** 2 for x in data) / n

            self.mean = float(mu)
            self.stddev = float(var ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of x."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of z."""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the PDF value for a given x-value.
        """
        pi = 3.141592653589793
        e = 2.718281828459045

        z = (x - self.mean) / self.stddev
        return (1 / (self.stddev * (2 * pi) ** 0.5)) * (e ** (-(z ** 2) / 2))

    def cdf(self, x):
        """
        Calculates the CDF value for a given x-value.
        """
        # CDF(x) = 0.5 * (1 + erf((x - mean) / (stddev * sqrt(2))))
        sqrt2 = 2 ** 0.5
        t = (x - self.mean) / (self.stddev * sqrt2)

        # erf approximation (Abramowitz & Stegun, 7.1.26)
        # erf(t) ≈ sign * (1 - P(t) * exp(-t^2))
        pi = 3.141592653589793
        e = 2.718281828459045

        sign = 1
        if t < 0:
            sign = -1
            t = -t

        # coefficients
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        u = 1.0 / (1.0 + p * t)
        poly = (((((a5 * u + a4) * u + a3) * u + a2) * u + a1) * u)

        erf_approx = 1.0 - poly * (e ** (-(t * t)))
        erf_approx *= sign

        return 0.5 * (1.0 + erf_approx)
