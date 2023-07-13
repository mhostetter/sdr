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
        dtype: The data type of the output array. If `None`, the smallest unsigned integer dtype
            that can hold `bpe` bits is used.

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

    if dtype is None:
        dtype = _unsigned_dtype(bpe)

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
        single_bits = 2 ** np.arange(bpe - 1, -1, -1, dtype=dtype)
        y = np.sum(X * single_bits, axis=-1)

    return y.astype(dtype)


@export
def unpack(x: npt.ArrayLike, bpe: int, dtype: npt.DTypeLike = np.uint8) -> np.ndarray:
    """
    Unpacks an array with multiple bits per element into a binary array.

    The data is assumed to have the most significant bit first.

    Arguments:
        x: The input array with `bpe` bits per element.
        bpe: The number of bits per element in the input array.
        dtype: The data type of the output array.

    Returns:
        The unpacked binary data with 1 bit per element.

    Examples:
        .. ipython:: python

            sdr.unpack([2, 0, 3, 1], 2)
            sdr.unpack([4, 3, 2], 3)

    Group:
        data
    """
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError("Argument 'x' must be an integer array, not have {x.dtype} dtype.")
    if not bpe >= 1:
        raise ValueError("Argument 'bpe' must be at least 1, not {bpe}.")

    if dtype is None:
        dtype = np.uint8

    if bpe == 1:
        y = x
    elif bpe == 8:
        # The unpackbits() function requires unsigned dtypes.
        dtype = _unsigned_dtype(bpe)
        y = np.unpackbits(x.astype(dtype), axis=-1)
    else:
        single_bits = 2 ** np.arange(bpe - 1, -1, -1, dtype=np.int64)
        X = (x[..., np.newaxis] & single_bits) > 0
        shape = x.shape[:-1] + (x.shape[-1] * bpe,)
        y = X.reshape(shape)

    return y.astype(dtype)


def _unsigned_dtype(bits: int) -> np.integer:
    """
    Returns the smallest unsigned integer dtype that can hold the specified number of bits.
    """
    if bits <= 8:
        return np.uint8
    elif bits <= 16:
        return np.uint16
    elif bits <= 32:
        return np.uint32
    elif bits <= 64:
        return np.uint64
    elif bits <= 128:
        return np.uint128
    elif bits <= 256:
        return np.uint256
    else:
        raise ValueError(f"Cannot create unsigned integer dtype for {bits} bits.")
