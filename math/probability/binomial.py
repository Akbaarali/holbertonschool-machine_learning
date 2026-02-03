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
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / len(data)

            p_est = 1.0 - (var / mean)
            n_est = round(mean / p_est)
            p_est = mean / n_est

            self.n = int(n_est)
            self.p = float(p_est)

    def pmf(self, k):
        """
        Calculates the value of the PMF
        Args:
            k (int): number of "successes"
        Returns:
            float: PMF value for k
        """
        # if k is not an integer, convert it to an integer
        k = int(k)

        # if k is out of range, return 0
        if k < 0 or k > self.n:
            return 0

        # Compute nCk without imports:
        # nCk = n! / (k!(n-k)!)
        # Use multiplicative formula to avoid huge factorials.
        n = self.n

        # symmetry to reduce loops
        if k > n - k:
            k = n - k

        comb = 1
        i = 1
        while i <= k:
            comb = comb * (n - k + i) / i
            i += 1
        return comb * (self.p ** k) * ((1 - self.p) ** (n - k))
