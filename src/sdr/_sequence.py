"""
A module containing various bipolar and binary sequences.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from ._helper import export


@export
def barker(size: int, output: Literal["binary", "bipolar"] = "bipolar") -> np.ndarray:
    """
    Returns the Barker code/sequence of length $N$.

    Arguments:
        size: The length $N$ of the Barker code/sequence.
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

        Barker sequences have maximally-small autocorrelation sidelobes of +1 or -1.

        .. ipython:: python

            corr = np.correlate(seq, seq, mode="full"); corr
            lag = np.arange(-seq.size - 1, seq.size)

            @savefig sdr_barker_1.png
            plt.figure(figsize=(8, 4)); \
            plt.plot(lag, np.abs(corr)**2); \
            plt.xlabel("Lag"); \
            plt.ylabel("Power"); \
            plt.title("Autocorrelation of Length-13 Barker Sequence"); \
            plt.tight_layout();

    Group:
        sequences
    """
    if not isinstance(size, int):
        raise TypeError(f"Argument 'size' must be of type 'int', not {type(size)}.")

    if size == 1:
        code = np.array([1])
    elif size == 2:
        code = np.array([1, 0])
    elif size == 3:
        code = np.array([1, 1, 0])
    elif size == 4:
        code = np.array([1, 1, 0, 1])
    elif size == 5:
        code = np.array([1, 1, 1, 0, 1])
    elif size == 7:
        code = np.array([1, 1, 1, 0, 0, 1, 0])
    elif size == 11:
        code = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0])
    elif size == 13:
        code = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    else:
        raise ValueError(f"Barker sequence of length {size} does not exist.")

    if output == "binary":
        return code

    # Map binary Barker code to bipolar. The mapping is equivalent to BPSK: 0 -> 1, 1 -> -1.
    sequence = 1 - 2 * code
    sequence = sequence.astype(np.float64)

    return sequence
