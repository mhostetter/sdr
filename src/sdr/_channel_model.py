"""
A module containing various channel models.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._helper import export


@export
def bsc(x: npt.ArrayLike, p: float) -> np.ndarray:
    r"""
    Passes the binary input sequence $x$ through a binary symmetric channel (BSC)
    with transition probability $p$.

    Arguments:
        x: The input sequence $x \in \{0, 1\}$.
        p: The probability $p$ of a bit flip.

    Returns:
        The output sequence $y \in \{0, 1\}$.

    Examples:
        When 20 bits are passed through a BSC with transition probability $p=0.25$,
        roughly 5 bits should be flipped at the output.

        .. ipython:: python

            x = np.random.randint(0, 2, 20); x
            y = sdr.bsc(x, 0.25); y
            x == y

    Group:
        channel-model
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Argument 'p' must be between 0 and 1, not {p}.")

    x = np.asarray(x)
    flip = np.random.choice([0, 1], size=x.shape, p=[1 - p, p])
    y = x ^ flip

    return y if y.ndim > 0 else y.item()


@export
def bec(x: npt.ArrayLike, p: float) -> np.ndarray:
    r"""
    Passes the binary input sequence $x$ through a binary erasure channel (BEC)
    with erasure probability $p$.

    Arguments:
        x: The input sequence $x \in \{0, 1\}$.
        p: The probability $p$ of a bit erasure.

    Returns:
        The output sequence $y \in \{0, 1, e\}$. Erasures $e$ are represented by -1.

    Examples:
        When 20 bits are passed through a BEC with erasure probability $p=0.25$,
        roughly 5 bits should be erased at the output.

        .. ipython:: python

            x = np.random.randint(0, 2, 20); x
            y = sdr.bec(x, 0.25); y
            x == y

    Group:
        channel-model
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Argument 'p' must be between 0 and 1, not {p}.")

    x = np.asarray(x)
    y = np.where(np.random.rand(*x.shape) < p, -1, x)

    return y if y.ndim > 0 else y.item()
