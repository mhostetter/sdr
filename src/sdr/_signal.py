"""
A module containing functions for signal manipulation.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._helper import export


@export
def mix(x: npt.ArrayLike, freq: float = 0, phase: float = 0, sample_rate: float = 1) -> np.ndarray:
    r"""
    Mixes the time-domain signal $x[n]$ with a complex exponential.

    $$y[n] = x[n] \cdot \exp \left[ j (\frac{2 \pi f}{f_s} n + \phi) \right]$$

    Arguments:
        x: The time-domain signal $x[n]$.
        freq: The frequency $f$ of the complex exponential in Hz (or 1/samples if `sample_rate=1`).
        phase: The phase $\phi$ of the complex exponential in degrees.
        sample_rate: The sample rate $f_s$ of the signal.

    Returns:
        The mixed signal $y[n]$.

    Examples:
        Create a complex exponential with a frequency of 10 Hz and phase of 45 degrees.

        .. ipython:: python

            sample_rate = 1e3; \
            N = 100; \
            x = np.exp(1j * (2 * np.pi * 10 * np.arange(N) / sample_rate + np.pi/4))

            @savefig sdr_mix_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, sample_rate=sample_rate); \
            plt.title(r"Complex exponential with $f=10$ Hz and $\phi=45$ degrees"); \
            plt.tight_layout();

        Mix the signal to baseband by removing the frequency rotation and the phase offset.

        .. ipython:: python

            y = sdr.mix(x, freq=-10, phase=-45, sample_rate=sample_rate)

            @savefig sdr_mix_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(y, sample_rate=sample_rate); \
            plt.title(r"Baseband signal with $f=0$ Hz and $\phi=0$ degrees"); \
            plt.tight_layout();

    Group:
        dsp-signal-manipulation
    """
    x = np.asarray(x)

    if not isinstance(freq, (int, float)):
        raise TypeError(f"Argument 'freq' must be a number, not {type(freq)}.")

    if not isinstance(phase, (int, float)):
        raise TypeError(f"Argument 'phase' must be a number, not {type(phase)}.")

    if not isinstance(sample_rate, (int, float)):
        raise TypeError(f"Argument 'sample_rate' must be a number, not {type(sample_rate)}.")
    if not sample_rate > 0:
        raise ValueError(f"Argument 'sample_rate' must be positive, not {sample_rate}.")

    t = np.arange(len(x)) / sample_rate  # Time vector in seconds
    y = x * np.exp(1j * (2 * np.pi * freq * t + np.deg2rad(phase)))

    return y
