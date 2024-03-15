"""
A module for various filter helper functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.signal


def get_frequency_vector(
    freqs: int | float | npt.ArrayLike = 1024,
    sample_rate: float = 1.0,
    whole: bool = True,
    decades: int | None = None,
) -> npt.NDArray[np.float64]:
    if isinstance(freqs, int):
        # freqs represents the number of frequency points
        if whole:
            max_f = sample_rate
        else:
            max_f = sample_rate / 2

        if decades is None:
            f = np.linspace(0, max_f, freqs, endpoint=False)
        elif isinstance(decades, int):
            f = np.logspace(np.log10(max_f) - decades, np.log10(max_f), freqs, endpoint=False)
        else:
            raise TypeError(f"Argument 'decades' must be an integer, not {type(decades).__name__}.")
    else:
        # freqs represents the single frequency or multiple frequencies
        f = np.asarray(freqs, dtype=float)
        f = np.atleast_1d(f)
        if not f.ndim <= 1:
            raise ValueError(f"Argument 'freqs' must be 0-D or 1-D, not {f.ndim}-D.")

    return f


def frequency_response(
    b: npt.ArrayLike,
    a: npt.ArrayLike,
    freqs: int | float | npt.ArrayLike = 1024,
    sample_rate: float = 1.0,
    whole: bool = True,
    decades: int | None = None,
) -> Any:
    f = get_frequency_vector(freqs, sample_rate, whole, decades)
    f, H = scipy.signal.freqz(b, a, worN=f, fs=sample_rate)
    return f, H
