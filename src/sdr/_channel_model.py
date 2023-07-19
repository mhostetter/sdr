"""
A module containing various channel models.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._helper import export


@export
def bsc(x: npt.ArrayLike, p: float) -> np.ndarray:
    """
    Passes the binary input sequence $x$ through a binary symmetric channel (BSC)
    with transition probability $p$.

    Args:
        x: The input sequence $x$.
        p: The probability $p$ of a bit flip.

    Returns:
        The output sequence $y$.

    Examples:
        A BSC with transition probability $p=0.5$ is equivalent to a random bit
        flip. For example, when 20 zeros are inputted, roughly 10 ones should
        appear in the output.

        .. ipython:: python

            sdr.bsc([0]*20, 0.5)

    Group:
        channel-model
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Argument 'p' must be between 0 and 1, not {p}.")

    x = np.asarray(x)
    flip = np.random.choice([0, 1], size=x.shape, p=[1 - p, p])
    y = x ^ flip

    return y if y.ndim > 0 else y.item()
