#!/usr/bin/env python3
"""
Normal distribution module
"""


class Normal:
    """Class representing a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor

        Args:
            data (list): list of data to calculate mean and stddev from
            mean (float): mean of the distribution
            stddev (float): standard deviation of the distribution
        """
        if data is None:
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """Calculates the z-score of x"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of z"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        pi = 3.141592653589793
        e = 2.718281828459045

        coeff = 1 / (self.stddev * (2 * pi) ** 0.5)
        expo = e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)
        return coeff * expo

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        (No imports allowed: uses erf approximation)
        """
        e = 2.718281828459045

        # Convert to z for erf argument: (x - mean) / (stddev * sqrt(2))
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Abramowitz & Stegun approximation for erf
        # erf(z) ≈ sign * [1 - (poly(t) * e^(-z^2))]
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        sign = 1
        if z < 0:
            sign = -1
            z = -z

        t = 1 / (1 + p * z)
        poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)

        erf = 1 - poly * (e ** (-z * z))
        erf = sign * erf

        return 0.5 * (1 + erf)
