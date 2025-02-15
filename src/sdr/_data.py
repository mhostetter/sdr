"""
A module containing functions for data manipulation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._helper import convert_output, export, verify_arraylike, verify_scalar


@export
def pack(
    x: npt.ArrayLike,
    bpe: int,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.int_]:
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
    x = verify_arraylike(x, int=True, atleast_1d=True)
    verify_scalar(bpe, int=True, positive=True)

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

    y = y.astype(dtype)

    return convert_output(y)


@export
def unpack(
    x: npt.ArrayLike,
    bpe: int,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray[np.int_]:
    """
    Unpacks an array with multiple bits per element into a binary array.

    The data is assumed to have the most significant bit first.

    Arguments:
        x: The input array with `bpe` bits per element.
        bpe: The number of bits per element in the input array.
        dtype: The data type of the output array. If `None`, :obj:`numpy.uint8` is used.

    Returns:
        The unpacked binary data with 1 bit per element.

    Examples:
        .. ipython:: python

            sdr.unpack([2, 0, 3, 1], 2)
            sdr.unpack([4, 3, 2], 3)

    Group:
        data
    """
    x = verify_arraylike(x, int=True, atleast_1d=True)
    verify_scalar(bpe, int=True, positive=True)

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

    y = y.astype(dtype)

    return convert_output(y)


@export
def hexdump(
    data: npt.ArrayLike | bytes,
    width: int = 16,
) -> str:
    """
    Returns a hexdump of the specified data.

    Arguments:
        data: The data to display. Each element is considered one byte. Use :func:`sdr.pack()`
            or :func:`sdr.unpack` to convert data with variable bits per element.
        width: The number of bytes per line.

    Returns:
        A string containing the hexdump of the data.

    Examples:
        .. ipython:: python

            print(sdr.hexdump(b"The quick brown fox jumps over the lazy dog"))

        .. ipython:: python

            print(sdr.hexdump([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], width=4))

    Group:
        data
    """
    if isinstance(data, bytes):
        data = np.frombuffer(data, dtype=np.uint8)
    else:
        data = verify_arraylike(data, int=True, ndim=1, inclusive_min=0, exclusive_max=256)
    verify_scalar(width, int=True, inclusive_min=1, inclusive_max=16)

    width = min(width, data.size)

    string = ""
    for i in range(0, data.size, width):
        line = data[i : i + width]
        string += f"{i:08x}  "
        string += " ".join(f"{x:02x}" for x in line)
        string += "   " * (width - line.size)
        string += "  "
        string += "".join(chr(x) if 32 <= x <= 126 else "." for x in line)
        string += "\n"

    return string


def _unsigned_dtype(bits: int) -> np.integer:
    """
    Returns the smallest unsigned integer dtype that can hold the specified number of bits.
    """
    if bits <= 8:
        return np.uint8
    if bits <= 16:
        return np.uint16
    if bits <= 32:
        return np.uint32
    if bits <= 64:
        return np.uint64
    if bits <= 128:
        return np.uint128
    if bits <= 256:
        return np.uint256

    raise ValueError(f"Cannot create unsigned integer dtype for {bits} bits.")
