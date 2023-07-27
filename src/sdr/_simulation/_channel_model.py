"""
A module containing various channel models.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export


@export
def bsc(x: npt.ArrayLike, p: float) -> np.ndarray:
    r"""
    Passes the binary input sequence $x$ through a binary symmetric channel (BSC)
    with transition probability $p$.

    Arguments:
        x: The input sequence $x$ with $x_i \in \{0, 1\}$.
        p: The probability $p$ of a bit flip.

    Returns:
        The output sequence $y$ with $y_i \in \{0, 1\}$.

    Examples:
        When 20 bits are passed through a BSC with transition probability $p=0.25$,
        roughly 5 bits are flipped at the output.

        .. ipython:: python

            x = np.random.randint(0, 2, 20); x
            y = sdr.bsc(x, 0.25); y
            x == y

    Group:
        simulation-channel-models
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
        x: The input sequence $x$ with $x_i \in \{0, 1\}$.
        p: The probability $p$ of a bit erasure.

    Returns:
        The output sequence $y$ with $y_i \in \{0, 1, e\}$. Erasures $e$ are represented by -1.

    Examples:
        When 20 bits are passed through a BEC with erasure probability $p=0.25$,
        roughly 5 bits are erased at the output.

        .. ipython:: python

            x = np.random.randint(0, 2, 20); x
            y = sdr.bec(x, 0.25); y
            x == y

    Group:
        simulation-channel-models
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Argument 'p' must be between 0 and 1, not {p}.")

    x = np.asarray(x)
    y = np.where(np.random.rand(*x.shape) < p, -1, x)

    return y if y.ndim > 0 else y.item()


@export
def dmc(
    x: npt.ArrayLike, P: npt.ArrayLike, X: npt.ArrayLike | None = None, Y: npt.ArrayLike | None = None
) -> np.ndarray:
    r"""
    Passes the input sequence $x$ through a discrete memoryless channel (DMC) with transition
    probability matrix $P$.

    Arguments:
        x: The input sequence $x$ with $x_i \in \mathcal{X}$.
        P: The $m \times n$ transition probability matrix $P$, where $P_{i,j} = \Pr(Y = y_j | X = x_i)$.
        X: The input alphabet $\mathcal{X}$ of size $m$. If `None`, it is assumed that
            $\mathcal{X} = \{0, 1, \ldots, m-1\}$.
        Y: The output alphabet $\mathcal{Y}$ of size $n$. If `None`, it is assumed that
            $\mathcal{Y} = \{0, 1, \ldots, n-1\}$.

    Returns:
        The output sequence $y$ with $y_i \in \mathcal{Y}$.

    Examples:
        Define the binary symmetric channel (BSC) with transition probability $p=0.25$.

        .. ipython:: python

            p = 0.25
            x = np.random.randint(0, 2, 20); x
            P = [[1 - p, p], [p, 1 - p]]
            y = sdr.dmc(x, P); y
            x == y

        Define the binary erasure channel (BEC) with erasure probability $p=0.25$.

        .. ipython:: python

            p = 0.25
            x = np.random.randint(0, 2, 20); x
            P = [[1 - p, 0, p], [0, 1 - p, p]]
            # Specify the erasure as -1
            y = sdr.dmc(x, P, Y=[0, 1, -1]); y
            x == y

    Group:
        simulation-channel-models
    """
    x = np.asarray(x)

    P = np.asarray(P)
    if not P.ndim == 2:
        raise ValueError(f"Argument 'P' must be a 2D array, not {P.ndim}D.")
    if not np.all(P >= 0):
        raise ValueError(f"Argument 'P' must have non-negative entires, not {P}.")
    if not np.allclose(P.sum(axis=1), 1):
        raise ValueError(f"Argument 'P' must have row that sum to 1, not {P.sum(axis=1)}.")

    X = np.asarray(X) if X is not None else np.arange(P.shape[0])
    if not X.ndim == 1:
        raise ValueError(f"Argument 'X' must be a 1D array, not {X.ndim}D.")
    if not P.shape[0] == X.shape[0]:
        raise ValueError(f"Argument 'P' must have {X.shape[0]} rows, not {P.shape[0]}.")

    Y = np.asarray(Y) if Y is not None else np.arange(P.shape[1])
    if not Y.ndim == 1:
        raise ValueError(f"Argument 'Y' must be a 1D array, not {Y.ndim}D.")
    if not P.shape[1] == Y.shape[0]:
        raise ValueError(f"Argument 'P' must have {Y.shape[0]} columns, not {P.shape[1]}.")

    y = np.zeros_like(x)
    for i, Xi in enumerate(X):
        idxs = np.where(x == Xi)[0]
        y[idxs] = np.random.choice(Y, size=idxs.shape, p=P[i])

    return y if y.ndim > 0 else y.item()
