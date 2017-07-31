import numpy as np


class Significance(object):
    """
    Implements equations in Li&Ma 1983

    """

    def __init__(self, Non, Noff, alpha=1):

        assert alpha > 0 and alpha <= 1, 'alpha was %f' % alpha

        self.Non = np.array(Non, dtype=float, ndmin=1)

        self.Noff = np.array(Noff, dtype=float, ndmin=1)

        self.alpha = float(alpha)

        self.expected = self.alpha * self.Noff

        self.net = self.Non - self.expected

    def li_and_ma_equivalent_for_gaussian_background(self, sigma_b):

        # This is a computation I need to publish (G. Vianello)

        b = self.expected
        o = self.Non

        b0 = 0.5 * (np.sqrt(b**2 - 2 * sigma_b**2 * (b - 2 * o) + sigma_b**4) +
                    b - sigma_b**2)

        S = np.sqrt(2) * np.sqrt(o * np.log(o / b0) + (b0 - b)**2 / (
            2 * sigma_b**2) + b0 - o)

        sign = np.where(o > b, 1, -1)

        return sign * S
