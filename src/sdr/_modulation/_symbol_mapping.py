"""
A module containing functions for various symbol mapping schemes.
"""
from __future__ import annotations

import numpy as np

from .._helper import export


@export
def binary_code(n: int) -> np.ndarray:
    """
    Generates a binary code of length $2^n$.

    Arguments:
        n: The length of the binary code.

    Returns:
        A binary code of length $2^n$.

    Examples:
        .. ipython:: python

            sdr.binary_code(1)
            sdr.binary_code(2)
            sdr.binary_code(3)
            sdr.binary_code(4)

    Group:
        modulation-symbol-mapping
    """
    if not n >= 1:
        raise ValueError(f"Argument 'n' must be greater than or equal to 1, not {n}.")

    return np.arange(2**n)


@export
def gray_code(n: int) -> np.ndarray:
    """
    Generates a Gray code of length $2^n$.

    Arguments:
        n: The length of the Gray code.

    Returns:
        A Gray code of length $2^n$.

    Examples:
        .. ipython:: python

            sdr.gray_code(1)
            sdr.gray_code(2)
            sdr.gray_code(3)
            sdr.gray_code(4)

    Group:
        modulation-symbol-mapping
    """
    if not n >= 1:
        raise ValueError(f"Argument 'n' must be greater than or equal to 1, not {n}.")

    if n == 1:
        return np.array([0, 1])

    # Generate the Gray code for n - 1.
    n1 = gray_code(n - 1)

    # Generate the Gray code for n by concatenating the Gray code for n - 1
    # with itself reversed. Also, the most significant bit of the second half
    # is set to 1.
    return np.concatenate((n1, n1[::-1] + 2 ** (n - 1)))
