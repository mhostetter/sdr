"""
A module containing functions for various symbol mapping schemes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export


@export
def binary_code(degree: int) -> npt.NDArray[np.int_]:
    """
    Generates a binary code of length $n = 2^m$.

    Arguments:
        degree: The degree $m$ of the binary code.

    Returns:
        A binary code of length $n = 2^m$.

    Examples:
        .. ipython:: python

            sdr.binary_code(1)
            sdr.binary_code(2)
            sdr.binary_code(3)
            sdr.binary_code(4)

    Group:
        sequences-symbol-mapping
    """
    if not degree >= 1:
        raise ValueError(f"Argument 'degree' must be greater than or equal to 1, not {degree}.")

    return np.arange(2**degree)


@export
def gray_code(degree: int) -> npt.NDArray[np.int_]:
    """
    Generates a Gray code of length $n = 2^m$.

    Arguments:
        degree: The degree $m$ of the Gray code.

    Returns:
        A Gray code of length $n = 2^m$.

    Examples:
        .. ipython:: python

            sdr.gray_code(1)
            sdr.gray_code(2)
            sdr.gray_code(3)
            sdr.gray_code(4)

    Group:
        sequences-symbol-mapping
    """
    if not degree >= 1:
        raise ValueError(f"Argument 'degree' must be greater than or equal to 1, not {degree}.")

    if degree == 1:
        return np.array([0, 1])

    # Generate the Gray code with degree m - 1
    code_1 = gray_code(degree - 1)

    # Generate the Gray code for degree m by concatenating the Gray code for degree m - 1
    # with itself reversed. Also, the most significant bit of the second half
    # is set to 1.
    return np.concatenate((code_1, code_1[::-1] + 2 ** (degree - 1)))
