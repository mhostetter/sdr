"""
A module containing functions for various symbol coding schemes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import convert_output, export, verify_arraylike, verify_scalar


@export
def diff_encode(
    x: npt.ArrayLike,
    y_prev: int = 0,
) -> npt.NDArray[np.int_]:
    r"""
    Differentially encodes the input data $x[k]$.

    $$y[k] = x[k] \oplus y[k-1]$$

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
    x = verify_arraylike(x, int=True, atleast_1d=True, ndim=1, non_negative=True)
    verify_scalar(y_prev, int=True, non_negative=True)

    y = np.empty_like(x, dtype=int)
    y[0] = x[0] ^ y_prev
    for n in range(1, len(x)):
        y[n] = x[n] ^ y[n - 1]

    return convert_output(y)


@export
def diff_decode(
    y: npt.ArrayLike,
    y_prev: int = 0,
) -> npt.NDArray[np.int_]:
    r"""
    Differentially decodes the input data $y[k]$.

    $$x[k] = y[k] \oplus y[k-1]$$

    Arguments:
        y: The input encoded data $y[k]$.
        y_prev: The previous value of the encoded data $y[k-1]$.

    Returns:
        The differentially decoded data $x[k]$.

    Notes:
        .. code-block:: text
            :caption: Differential Decoder Block Diagram

                       +------+  y[k-1]
            y[k] --+-->| z^-1 |--------->@--> x[k]
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
    y = verify_arraylike(y, int=True, atleast_1d=True, ndim=1, non_negative=True)
    verify_scalar(y_prev, int=True, non_negative=True)

    x = np.empty_like(y, dtype=int)
    x[0] = y[0] ^ y_prev
    for n in range(1, len(y)):
        x[n] = y[n] ^ y[n - 1]

    return convert_output(x)
