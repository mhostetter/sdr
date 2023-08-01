"""
A module containing functions for various symbol coding schemes.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export


@export
def diff_encode(x: npt.ArrayLike, y_prev: int = 0) -> np.ndarray:
    """
    Differentially encodes the input data $x[k]$.

    Args:
        x: The input uncoded data $x[k]$.
        y_prev: The previous value of the output encoded data $y[k-1]$.

    Returns:
        The differentially encoded data $y[k]$.

    Notes:
        .. code-block:: text
            :caption: Differential Encoder Block Diagram

            x[k] -->@--------------+--> y[k]
                    ^              |
                    |   +------+   |
            y[k-1]  +---| z^-1 |<--+
                        +------+

            x[k] = Input data
            y[k] = Encoded data
            z^-1 = Unit delay
            @ = Adder

    Examples:
        .. ipython:: python

            sdr.diff_encode([0, 1, 0, 0, 1, 1])
            sdr.diff_decode([0, 1, 1, 1, 0, 1])

    Group:
        modulation-symbol-encoding
    """
    x = np.asarray(x)
    if not x.ndim == 1:
        raise ValueError(f"Argument 'x' must be a 1D array, not {x.ndim}D.")
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError(f"Argument 'x' must be an integer array, not {x.dtype}.")

    if not isinstance(y_prev, int):
        raise TypeError(f"Argument 'y_prev' must be an integer, not {type(y_prev)}.")
    if not y_prev >= 0:
        raise ValueError(f"Argument 'y_prev' must be non-negative, not {y_prev}.")

    y = np.empty_like(x, dtype=int)
    y[0] = x[0] ^ y_prev
    for n in range(1, len(x)):
        y[n] = x[n] ^ y[n - 1]

    return y


@export
def diff_decode(y: npt.ArrayLike, y_prev: int = 0) -> np.ndarray:
    """
    Differentially decodes the input data $y[k]$.

    Arguments:
        y: The input encoded data $y[k]$.
        y_prev: The previous value of the encoded data $y[k-1]$.

    Returns:
        The differentially decoded data $x[k]$.

    Notes:
        .. code-block:: text
            :caption: Differential Decoder Block Diagram

                       +------+  y[k-1]
            y[k] --+-->| z^-1 |----------@--> x[k]
                   |   +------+          ^
                   |                     |
                   +---------------------+

            y[k] = Encoded data
            x[k] = Decoded data
            z^-1 = Unit delay
            @ = Adder

    Examples:
        .. ipython:: python

            sdr.diff_decode([0, 1, 1, 1, 0, 1])
            sdr.diff_encode([0, 1, 0, 0, 1, 1])

    Group:
        modulation-symbol-encoding
    """
    y = np.asarray(y)
    if not y.ndim == 1:
        raise ValueError(f"Argument 'y' must be a 1D array, not {y.ndim}D.")
    if not np.issubdtype(y.dtype, np.integer):
        raise TypeError(f"Argument 'y' must be an integer array, not {y.dtype}.")

    if not isinstance(y_prev, int):
        raise TypeError(f"Argument 'y_prev' must be an integer, not {type(y_prev)}.")
    if not y_prev >= 0:
        raise ValueError(f"Argument 'y_prev' must be non-negative, not {y_prev}.")

    x = np.empty_like(y, dtype=int)
    x[0] = y[0] ^ y_prev
    for n in range(1, len(y)):
        x[n] = y[n] ^ y[n - 1]

    return x
