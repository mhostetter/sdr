"""
A module containing functions for data manipulation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from ._helper import export


@export
def pack(x: npt.ArrayLike, bpe: int, dtype: Optional[npt.DTypeLike] = None) -> np.ndarray:
    """
    Packs a binary array into an array with multiple bits per element.

    The data is assumed to have the most significant bit first. If there are not enough bits in the
    input array to fill the last element of the output array, the remaining bits are filled with
    zeros.

    Arguments:
        x: The input binary array with 1 bit per element.
        bpe: The number of bits per element in the output array.
        dtype: The data type of the output array. If `None`, the data type of the input array is used
            (assuming it can hold `bpe` bits per element), otherwise :obj:`np.int64` is used.

    Returns:
        The packed data with `bpe` bits per element.

    Examples:
        .. ipython:: python

            sdr.pack([1, 0, 0, 0, 1, 1, 0, 1], 2)
            sdr.pack([1, 0, 0, 0, 1, 1, 0, 1], 3)

    Group:
        data
    """
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError("Argument 'x' must be an integer array, not have {x.dtype} dtype.")
    if not bpe >= 1:
        raise ValueError("Argument 'bpe' must be at least 1, not {bpe}.")

    if bpe == 1:
        y = x
    elif bpe == 8:
        y = np.packbits(x, axis=-1)
    else:
        if x.shape[-1] % bpe != 0:
            # Pad the last axis
            pad = bpe - (x.shape[-1] % bpe)
            pad_width = ((0, 0),) * (x.ndim - 1) + ((0, pad),)
            x = np.pad(x, pad_width, mode="constant", constant_values=0)

        shape = x.shape[:-1] + (x.shape[-1] // bpe, bpe)
        X = x.reshape(shape)
        scalars = 2 ** np.arange(bpe - 1, -1, -1, dtype=np.int64)
        y = np.sum(X * scalars, axis=-1)

    if dtype is None:
        if np.iinfo(x.dtype).max >= 2**bpe:
            dtype = x.dtype
        else:
            dtype = np.int64

    return y.astype(dtype)
