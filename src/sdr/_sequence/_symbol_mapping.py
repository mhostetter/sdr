"""
A module containing functions for various symbol mapping schemes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export, verify_scalar


@export
def binary_code(length: int) -> npt.NDArray[np.int_]:
    """
    Generates a binary code of length $n = 2^m$.

    Arguments:
        length: The length $n = 2^m$ of the binary code.

    Returns:
        A binary code of length $n = 2^m$.

    Examples:
        .. ipython:: python

            sdr.binary_code(2)
            sdr.binary_code(4)
            sdr.binary_code(8)
            sdr.binary_code(16)

    Group:
        sequences-symbol-mapping
    """
    verify_scalar(length, int=True, positive=True, power_of_two=True)

    return np.arange(length)


@export
def gray_code(length: int) -> npt.NDArray[np.int_]:
    """
    Generates a Gray code of length $n = 2^m$.

    Arguments:
        length: The length $n = 2^m$ of the Gray code.

    Returns:
        A Gray code of length $n = 2^m$.

    Examples:
        .. ipython:: python

            sdr.gray_code(2)
            sdr.gray_code(4)
            sdr.gray_code(8)
            sdr.gray_code(16)

    Group:
        sequences-symbol-mapping
    """
    verify_scalar(length, int=True, positive=True, power_of_two=True)

    if length == 2:
        return np.array([0, 1])

    # Generate the Gray code with degree m - 1
    code_1 = gray_code(length // 2)

    # Generate the Gray code for degree m by concatenating the Gray code for degree m - 1
    # with itself reversed. Also, the most significant bit of the second half
    # is set to 1.
    return np.concatenate((code_1, code_1[::-1] + length // 2))
