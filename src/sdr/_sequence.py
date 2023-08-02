"""
A module containing various bipolar and binary sequences.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
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

            corr = np.correlate(seq, seq, mode="full"); corr
            lag = np.arange(-seq.size + 1, seq.size)

            @savefig sdr_barker_1.png
            plt.figure(figsize=(8, 4)); \
            plt.plot(lag, np.abs(corr)); \
            plt.xlabel("Lag"); \
            plt.ylabel("Magnitude"); \
            plt.title("Autocorrelation of Length-13 Barker Sequence"); \
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
    sequence = sequence.astype(np.float64)

    return sequence
