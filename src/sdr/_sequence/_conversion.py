"""
A module containing conversions between binary and bipolar sequences.
"""
from __future__ import annotations

import galois
import numpy as np
import numpy.typing as npt


def code_to_sequence(code: npt.NDArray[np.int_]) -> npt.NDArray[np.float_]:
    r"""
    Converts a binary code to a bipolar sequence.

    Notes:
        The mapping is equivalent to BPSK: 0 -> 1, 1 -> -1.
    """
    sequence = 1 - 2 * code
    sequence = sequence.astype(float)
    return sequence


def code_to_field(code: npt.NDArray[np.int_]) -> galois.FieldArray:
    r"""
    Converts a binary code to a Galois field array over $\mathrm{GF}(2)$.
    """
    GF = galois.GF(2)
    field = GF(code)
    return field


def sequence_to_code(sequence: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:
    r"""
    Converts a bipolar sequence to a binary code.

    Notes:
        The mapping is equivalent to BPSK: 1 -> 0, -1 -> 1.
    """
    code = (1 - sequence) / 2
    code = code.astype(int)
    return code
