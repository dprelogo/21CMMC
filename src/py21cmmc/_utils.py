"""Utility functions used throughout 21CMMC."""

try:
    from collections import Iterable  # Python <= 3.9

except ImportError:
    from collections.abc import Iterable  # Python > 3.9

import numpy as np
from scipy.special import erf, erfinv


def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def phi(x):
    """Integral of the unit-variance gaussian."""
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def phiinv(x):
    """Inverse of the integral of the unit-variance gaussian."""
    return erfinv(2 * x + 1)
