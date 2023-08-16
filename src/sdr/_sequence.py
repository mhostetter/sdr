"""
A module containing various bipolar and binary sequences.
"""
from __future__ import annotations

import math

import numpy as np
from typing_extensions import Literal

from ._helper import export


@export
def barker(length: int, output: Literal["binary", "bipolar"] = "bipolar") -> np.ndarray:
    r"""
    Returns the Barker code/sequence of length $N$.

    Arguments:
        length: The length $N$ of the Barker code/sequence.
        output: The output format of the Barker code/sequence.

            - `"binary"`: The Barker code with binary values of 0 and 1.
            - `"bipolar"`: The Barker sequence with bipolar values of 1 and -1.

    Returns:
        The Barker code/sequence of length $N$.

    Examples:
        Create a Barker code and sequence of length 13.

        .. ipython:: python

            code = sdr.barker(13, output="binary"); code
            seq = sdr.barker(13); seq

        Barker sequences have ideally-minimal autocorrelation sidelobes of +1 or -1.

        .. ipython:: python

            corr = np.correlate(seq, seq, mode="full"); \
            lag = np.arange(-seq.size + 1, seq.size)

            @savefig sdr_barker_1.png
            plt.figure(figsize=(8, 4)); \
            plt.plot(lag, np.abs(corr)); \
            plt.xlabel("Lag"); \
            plt.ylabel("Magnitude"); \
            plt.title("Autocorrelation of length-13 Barker sequence"); \
            plt.tight_layout();

    Group:
        sequences
    """
    if not isinstance(length, int):
        raise TypeError(f"Argument 'length' must be of type 'int', not {type(length)}.")

    if length == 1:
        code = np.array([1])
    elif length == 2:
        code = np.array([1, 0])
    elif length == 3:
        code = np.array([1, 1, 0])
    elif length == 4:
        code = np.array([1, 1, 0, 1])
    elif length == 5:
        code = np.array([1, 1, 1, 0, 1])
    elif length == 7:
        code = np.array([1, 1, 1, 0, 0, 1, 0])
    elif length == 11:
        code = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0])
    elif length == 13:
        code = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    else:
        raise ValueError(f"Barker sequence of length {length} does not exist.")

    if output == "binary":
        return code

    # Map binary Barker code to bipolar. The mapping is equivalent to BPSK: 0 -> 1, 1 -> -1.
    sequence = 1 - 2 * code
    sequence = sequence.astype(float)

    return sequence


@export
def zadoff_chu(length: int, root: int, shift: int = 0) -> np.ndarray:
    r"""
    Returns the root-$u$ Zadoff-Chu sequence of length $N$.

    Arguments:
        length: The length $N$ of the Zadoff-Chu sequence.
        root: The root $0 < u < N$ of the Zadoff-Chu sequence. The root must be relatively prime to the length,
            i.e., $\gcd(u, N) = 1$.
        shift: The shift $q \in \mathbb{Z}$ of the Zadoff-Chu sequence. When $q \ne 0$, the returned sequence
            is a cyclic shift of the root-$u$ Zadoff-Chu sequence.

    Returns:
        The root-$u$ Zadoff-Chu sequence of length $N$.

    Notes:
        The root-$u$ Zadoff-Chu sequence with length $N$ and shift $q$ is defined as

        $$x_u[n] = \exp \left( -j \frac{\pi u n (n + c_{f} + 2q)}{N} \right) ,$$

        where $c_{f} = N \mod 2$.

    References:
        - https://en.wikipedia.org/wiki/Zadoff%E2%80%93Chu_sequence

    Examples:
        Create a root-3 Zadoff-Chu sequence $x_3[n]$ with length 139.

        .. ipython:: python

            N = 139; \
            x3 = sdr.zadoff_chu(N, 3)

            @savefig sdr_zadoff_chu_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(x3, linestyle="-", linewidth=0.5); \
            plt.title(f"Root-3 Zadoff-Chu sequence of length {N}"); \
            plt.tight_layout();

        The *periodic* autocorrelation of a Zadoff-Chu sequence has sidelobes of magnitude 0.

        .. ipython:: python

            # Perform periodic autocorrelation
            corr = np.correlate(np.roll(np.tile(x3, 2), -N//2), x3, mode="valid"); \
            lag = np.arange(-N//2 + 1, N//2 + 2)

            @savefig sdr_zadoff_chu_2.png
            plt.figure(figsize=(8, 4)); \
            plt.plot(lag, np.abs(corr) / N); \
            plt.ylim(0, 1); \
            plt.xlabel("Lag"); \
            plt.ylabel("Magnitude"); \
            plt.title(f"Periodic autocorrelation of root-3 Zadoff-Chu sequence of length {N}"); \
            plt.tight_layout();

        Create a root-5 Zadoff-Chu sequence $x_5[n]$ with length 139.

        .. ipython:: python

            x5 = sdr.zadoff_chu(N, 5)

            @savefig sdr_zadoff_chu_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(x5, linestyle="-", linewidth=0.5); \
            plt.title(f"Root-5 Zadoff-Chu sequence of length {N}"); \
            plt.tight_layout();

        The *periodic* cross correlation of two prime-length Zadoff-Chu sequences with different roots has sidelobes
        with magnitude $1 / \sqrt{N}$.

        .. ipython:: python

            # Perform periodic cross correlation
            xcorr = np.correlate(np.roll(np.tile(x3, 2), -N//2), x5, mode="valid"); \
            lag = np.arange(-N//2 + 1, N//2 + 2)

            @savefig sdr_zadoff_chu_4.png
            plt.figure(figsize=(8, 4)); \
            plt.plot(lag, np.abs(xcorr) / N); \
            plt.ylim(0, 1); \
            plt.xlabel("Lag"); \
            plt.ylabel("Magnitude"); \
            plt.title(f"Periodic cross correlation of root-3 and root-5 Zadoff-Chu sequences of length {N}"); \
            plt.tight_layout();

    Group:
        sequences
    """
    if not isinstance(length, int):
        raise TypeError(f"Argument 'length' must be of type 'int', not {type(length)}.")
    if not length > 1:
        raise ValueError(f"Argument 'length' must be greater than 1, not {length}.")

    if not isinstance(root, int):
        raise TypeError(f"Argument 'root' must be of type 'int', not {type(root)}.")
    if not 0 < root < length:
        raise ValueError(f"Argument 'root' must be greater than 0 and less than 'length', not {root}.")
    if not math.gcd(length, root) == 1:
        raise ValueError(f"Argument 'root' must be relatively prime to 'length'; {root} and {length} are not.")

    if not isinstance(shift, int):
        raise TypeError(f"Argument 'shift' must be of type 'int', not {type(shift)}.")

    n = np.arange(length)
    cf = length % 2
    xu = np.exp(-1j * np.pi * root * n * (n + cf + 2 * shift) / length)

    return xu
