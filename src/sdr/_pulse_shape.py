"""
A module containing various discrete-time pulse shapes.
"""
from __future__ import annotations

import numpy as np

from ._helper import export


@export
def root_raised_cosine(alpha: float, sps: int, N_symbols: int) -> np.ndarray:
    r"""
    Returns a square root raised cosine (SRRC) pulse shape.

    Arguments:
        alpha: The excess bandwidth $0 \le \alpha \le 1$ of the filter.
        sps: The number of samples per symbol.
        N_symbols: The length of the filter in symbols. The filter must have even order, `sps * N_symbols == 1`.
            The length of the filter is `sps * N_symbols + 1`.

    Returns:
        The root-raised cosine pulse shape with unit energy.

    References:
        * Michael Rice, *Digital Communications: A Discrete Time Approach*, Appendix A.

    Group:
        pulse-shape
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Argument 'alpha' must be between 0 and 1.")
    if not sps > 1:
        raise ValueError("Argument 'sps' must be greater than 1.")
    if not N_symbols > 1:
        raise ValueError("Argument 'N_symbols' must be greater than 1.")
    if not sps * N_symbols % 2 == 0:
        raise ValueError("The order of the filter (sps * N_symbols) must be even.")

    t = np.arange(-(sps * N_symbols) // 2, (sps * N_symbols) // 2 + 1, dtype=np.float32)
    Ts = sps  # Symbol duration (in samples)

    # Handle special cases where the denominator is zero
    t1 = Ts / (4 * alpha)
    t[t == -t1] -= 1e-4
    t[t == t1] += 1e-4
    t[t == 0] += 1e-8

    # Equation A-30
    A = np.sin(np.pi * (1 - alpha) * t / Ts)
    B = 4 * alpha * t / Ts * np.cos(np.pi * (1 + alpha) * t / Ts)
    C = np.pi * t / Ts * (1 - (4 * alpha * t / Ts) ** 2)
    h = (A + B) / C
    h /= 1 / np.sqrt(Ts)

    # Make the filter have unit energy
    h /= np.sqrt(np.sum(np.abs(h) ** 2))

    return h


@export
def raised_cosine(alpha: float, sps: int, N_symbols: int) -> np.ndarray:
    r"""
    Returns a raised cosine (RC) pulse shape.

    Arguments:
        alpha: The excess bandwidth $0 \le \alpha \le 1$ of the filter.
        sps: The number of samples per symbol.
        N_symbols: The length of the filter in symbols. The filter must have even order, `sps * N_symbols == 1`.
            The length of the filter is `sps * N_symbols + 1`.

    Returns:
        The raised cosine pulse shape with unit energy.

    References:
        * Michael Rice, *Digital Communications: A Discrete Time Approach*, Appendix A.

    Group:
        pulse-shape
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Argument 'alpha' must be between 0 and 1.")
    if not sps > 1:
        raise ValueError("Argument 'sps' must be greater than 1.")
    if not N_symbols > 1:
        raise ValueError("Argument 'N_symbols' must be greater than 1.")
    if not sps * N_symbols % 2 == 0:
        raise ValueError("The order of the filter (sps * N_symbols) must be even.")

    t = np.arange(-(sps * N_symbols) // 2, (sps * N_symbols) // 2 + 1, dtype=np.float32)
    Ts = sps

    # Handle special cases where the denominator is zero
    t1 = Ts / (2 * alpha)
    t[t == -t1] -= 1e-4
    t[t == t1] += 1e-4
    t[t == 0] += 1e-8

    # Equation A-27
    A = np.sin(np.pi * t / Ts) / (np.pi * t / Ts)
    B = np.cos(np.pi * alpha * t / Ts) / (1 - (2 * alpha * t / Ts) ** 2)
    h = A * B

    # Make the filter have unit energy
    h /= np.sqrt(np.sum(np.abs(h) ** 2))

    return h
