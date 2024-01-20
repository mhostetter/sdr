"""
A module containing various channel models.
"""
from __future__ import annotations

from typing import Any, overload

import numpy as np
import numpy.typing as npt

from .._helper import export
from .._link_budget._capacity import Hb


@export
def bsc(
    x: npt.ArrayLike,
    p: float,
    seed: int | None = None,
) -> npt.NDArray[np.int_]:
    r"""
    Passes the binary input sequence $x$ through a binary symmetric channel (BSC)
    with transition probability $p$.

    Arguments:
        x: The input sequence $x$ with $x_i \in \{0, 1\}$.
        p: The probability $p$ of a bit flip.
        seed: The seed for the random number generator. This is passed to :func:`numpy.random.default_rng()`.

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
    rng = np.random.default_rng(seed)
    flip = rng.choice([0, 1], size=x.shape, p=[1 - p, p])
    y = x ^ flip

    return y if y.ndim > 0 else y.item()


@export
def bec(
    x: npt.ArrayLike,
    p: float,
    seed: int | None = None,
) -> npt.NDArray[np.int_]:
    r"""
    Passes the binary input sequence $x$ through a binary erasure channel (BEC)
    with erasure probability $p$.

    Arguments:
        x: The input sequence $x$ with $x_i \in \{0, 1\}$.
        p: The probability $p$ of a bit erasure.
        seed: The seed for the random number generator. This is passed to :func:`numpy.random.default_rng()`.

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
    rng = np.random.default_rng(seed)
    random_p = rng.random(x.shape)
    y = np.where(random_p < p, -1, x)

    return y if y.ndim > 0 else y.item()


@export
def dmc(
    x: npt.ArrayLike,
    P: npt.ArrayLike,
    X: npt.ArrayLike | None = None,
    Y: npt.ArrayLike | None = None,
    seed: int | None = None,
) -> npt.NDArray[np.int_]:
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
        seed: The seed for the random number generator. This is passed to :func:`numpy.random.default_rng()`.

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
    rng = np.random.default_rng(seed)
    for i, Xi in enumerate(X):
        idxs = np.where(x == Xi)[0]
        y[idxs] = rng.choice(Y, size=idxs.shape, p=P[i])

    return y if y.ndim > 0 else y.item()


@export
class Channel:
    """
    A base class for wireless channels.

    Group:
        simulation-channel-models
    """

    def __init__(self, seed: int | None = None):
        """
        Creates a new channel.

        Arguments:
            seed: The seed for the random number generator. This is passed to :func:`numpy.random.default_rng()`.
        """
        self._rng: np.random.Generator  # Will be set in self.reset()

        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        """
        Resets the channel with a new seed.

        Arguments:
            seed: The seed for the random number generator. This is passed to :func:`numpy.random.default_rng()`.
        """
        self._rng = np.random.default_rng(seed)

    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        """
        Passes the input sequence $x$ through the channel.

        Arguments:
            x: The input sequence $x$.

        Returns:
            The output sequence $y$.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        return string

    @staticmethod
    def capacities() -> npt.NDArray:
        """
        Computes the channel capacity given the channel configuration.

        Returns:
            The channel capacities $C$ in bits/2D.
        """
        raise NotImplementedError

    @property
    def capacity(self) -> float:
        """
        The channel capacity $C$ in bits/2D of the instantiated channel.
        """
        raise NotImplementedError


@export
class BinarySymmetricChannel(Channel):
    r"""
    Implements a binary symmetric channel (BSC).

    Notes:
        The inputs to the BSC are $x_i \in \{0, 1\}$ and the outputs are $y_i \in \{0, 1\}$.
        The capacity of the BSC is

        $$C = 1 - H_b(p) \ \ \text{bits/channel use} .$$

    Examples:
        When 20 bits are passed through a BSC with transition probability $p=0.25$, roughly 5 bits are flipped
        at the output.

        .. ipython:: python

            bsc = sdr.BinarySymmetricChannel(0.25, seed=1)
            x = np.random.randint(0, 2, 20); x
            y = bsc(x); y
            np.count_nonzero(x != y)

        The capacity of this BSC is 0.189 bits/channel use.

        .. ipython:: python

            bsc.capacity

        When the probability $p$ of bit error is 0, the capacity of the channel is 1 bit/channel use.
        However, as the probability of bit error approaches 0.5, the capacity of the channel approaches
        0.

        .. ipython:: python

            p = np.linspace(0, 1, 100); \
            C = sdr.BinarySymmetricChannel.capacities(p)

            @savefig sdr_BinarySymmetricChannel_1.png
            plt.figure(figsize=(8, 4)); \
            plt.plot(p, C); \
            plt.xlabel("Transition probability, $p$"); \
            plt.ylabel("Capacity (bits/channel use), $C$"); \
            plt.title("Capacity of the Binary Symmetric Channel"); \
            plt.grid(True); \
            plt.tight_layout()

    Group:
        simulation-channel-models
    """

    def __init__(self, p: float, seed: int | None = None):
        """
        Creates a new binary symmetric channel (BSC).

        Arguments:
            p: The transition probability $p$ of the BSC channel.
            seed: The seed for the random number generator. This is passed to :func:`numpy.random.default_rng()`.
        """
        super().__init__(seed=seed)

        if not 0 <= p <= 1:
            raise ValueError(f"Argument 'p' must be between 0 and 1, not {p}.")
        self._p = p

    @overload
    def __call__(self, x: int) -> int:
        ...

    @overload
    def __call__(self, x: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        ...

    def __call__(self, x: Any) -> Any:
        r"""
        Passes the binary input sequence $x$ through the channel.

        Arguments:
            x: The input sequence $x$ with $x_i \in \{0, 1\}$.

        Returns:
            The output sequence $y$ with $y_i \in \{0, 1\}$.
        """
        x = np.asarray(x)
        flip = self._rng.choice([0, 1], size=x.shape, p=[1 - self.p, self.p])
        y = x ^ flip

        return y if y.ndim > 0 else y.item()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

    def __str__(self) -> str:
        string = super().__str__()
        string += f" p={self.p}"
        return string

    @staticmethod
    def capacities(p: npt.ArrayLike) -> npt.NDArray[np.float_]:
        """
        Calculates the capacity of BSC channels.

        Returns:
            The capacity $C$ of the channel in bits/channel use.
        """
        p = np.asarray(p)
        if not (np.all(0 <= p) and np.all(p <= 1)):
            raise ValueError(f"Argument 'p' must be between 0 and 1, not {p}.")

        return 1 - Hb(p)

    @property
    def capacity(self) -> float:
        """
        The capacity $C$ of the instantiated channel in bits/channel use.
        """
        return BinarySymmetricChannel.capacities(self.p)

    @property
    def p(self) -> float:
        """
        The transition probability $p$ of the BSC channel.
        """
        return self._p
