#!/usr/bin/env python3
"""
Normal distribution module
"""


class Normal:
    """Class representing a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
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

    def pdf(self, x):
        pi = 3.141592653589793
        return (
            1 / (self.stddev * (2 * pi) ** 0.5)
            * (2.718281828459045 ** (
                -0.5 * ((x - self.mean) / self.stddev) ** 2
            ))
        )

    def cdf(self, x):
        """
        Calculates the CDF of x without imports
        """
        # constants
        p = 0.3275911
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        sign = 1
        if z < 0:
            sign = -1
            z = -z

        t = 1 / (1 + p * z)

        erf = 1 - (
            (((((a5 * t + a4) * t + a3) * t + a2) * t + a1)
             * t)
            * (2.718281828459045 ** (-z * z))
        )

        return 0.5 * (1 + sign * erf)
