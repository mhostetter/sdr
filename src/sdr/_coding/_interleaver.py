"""
A module containing various interleavers.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export


@export
class Interleaver:
    r"""
    Implements an arbitrary interleaver.

    Examples:
        Create a length-4 interleaver with custom permutation map.

        .. ipython:: python

            pi = np.array([0, 3, 1, 2])
            interleaver = sdr.Interleaver(pi)
            interleaver.map
            interleaver.inverse_map

        Interleave and deinterleave a sequence.

        .. ipython:: python

            x = np.arange(16); x
            y = interleaver.interleave(x); y
            interleaver.deinterleave(y)

    Group:
        coding-interleavers
    """

    def __init__(self, map: npt.NDArray[np.int_]):
        r"""
        Creates an arbitrary interleaver.

        Arguments:
            map: The interleaver permutation map $\pi : i \mapsto j$, containing the values $[0, N)$.
                The $i$-th input element will be placed at the $\pi(i)$-th output position.
        """
        if not isinstance(map, np.ndarray):
            raise TypeError(f"Argument 'map' must be a NumPy array, not {type(map)}.")
        if not np.issubdtype(map.dtype, np.integer):
            raise TypeError(f"Argument 'map' must be an array of integers, not {map.dtype}.")
        if not np.all(np.unique(map) == np.arange(len(map))):
            raise ValueError(f"Argument 'map' must contain the integers [0, {len(map)}), not {map}.")

        self._map = map
        self._inverse_map = np.argsort(map)

    def __len__(self) -> int:
        """
        The size of the interleaver.
        """
        return self.map.size

    def interleave(self, x: npt.NDArray) -> npt.NDArray:
        r"""
        Interleaves the input sequence $x[n]$.

        Arguments:
            x: The input sequence $x[n]$. Length must be a multiple of the interleaver size.

        Returns:
            The interleaved sequence $y[n]$.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Argument `x` must be a NumPy array, not {type(x)}.")
        if not x.size % len(self) == 0:
            raise ValueError(f"Argument `x` must have a length that is a multiple of {len(self)}, not {x.size}.")

        y = np.empty_like(x)
        y.reshape((-1, len(self)))[..., self.map] = x.reshape((-1, len(self)))

        return y

    def deinterleave(self, y: npt.NDArray) -> npt.NDArray:
        r"""
        Deinterleaves the input sequence $y[n]$.

        Arguments:
            y: The input sequence $y[n]$. Length must be a multiple of the interleaver size.

        Returns:
            The deinterleaved sequence $x[n]$.
        """
        if not isinstance(y, np.ndarray):
            raise TypeError(f"Argument `y` must be a NumPy array, not {type(y)}.")
        if not y.size % len(self) == 0:
            raise ValueError(f"Argument `y` must have a length that is a multiple of {len(self)}, not {y.size}.")

        x = np.empty_like(y)
        x.reshape((-1, len(self)))[..., self.inverse_map] = y.reshape((-1, len(self)))

        return x

    @property
    def map(self) -> npt.NDArray[np.int_]:
        r"""
        The interleaver permutation map $\pi$.

        The map $\pi : i \mapsto j$ indicates that the $i$-th interleaver input will be placed at the $\pi(i)$-th
        output position.
        """
        return self._map

    @property
    def inverse_map(self) -> npt.NDArray[np.int_]:
        r"""
        The deinterleaver permutation map $\pi^{-1}$.

        The map $\pi^{-1} : j \mapsto i$ indicates that the $j$-th deinterleaver input will be placed at the
        $\pi^{-1}(j)$-th output position.
        """
        return self._inverse_map


@export
class BlockInterleaver(Interleaver):
    r"""
    Implements a block interleaver.

    Notes:
        A block interleaver feeds the input down the columns of a $R \times C$ matrix and then reads the output
        across the rows.

    Examples:
        Create a $3 \times 4$ block interleaver.

        .. ipython:: python

            interleaver = sdr.BlockInterleaver(3, 4)
            interleaver.map
            interleaver.inverse_map

        Interleave and deinterleave a sequence.

        .. ipython:: python

            x = np.arange(12); x
            y = interleaver.interleave(x); y
            interleaver.deinterleave(y)

    Group:
        coding-interleavers
    """

    def __init__(self, rows: int, cols: int):
        r"""
        Creates a $R \times C$ block interleaver.

        Arguments:
            rows: The number of rows $R$ in the interleaver. The row size determines the output separation of
                consecutive input elements.
            cols: The number of columns $C$ in the interleaver.
        """
        if not isinstance(rows, int):
            raise TypeError(f"Argument 'rows' must be an integer, not {type(rows)}.")
        if not isinstance(cols, int):
            raise TypeError(f"Argument 'cols' must be an integer, not {type(cols)}.")

        map = np.arange(rows * cols).reshape((rows, cols)).T.ravel()
        super().__init__(map)
