"""
A module containing various interleavers.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import convert_output, export, verify_arraylike, verify_equation, verify_scalar


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

    def __init__(self, map: npt.ArrayLike):
        r"""
        Creates an arbitrary interleaver.

        Arguments:
            map: The interleaver permutation map $\pi : i \mapsto j$, containing the values $[0, N)$.
                The $i$-th input element will be placed at the $\pi(i)$-th output position.
        """
        map = verify_arraylike(map, int=True, atleast_1d=True, ndim=1)
        verify_equation(np.unique(map).size == map.size)

        self._map = map
        self._inverse_map = np.argsort(map)

    def __len__(self) -> int:
        """
        The size of the interleaver.
        """
        return self.map.size

    def interleave(self, x: npt.ArrayLike) -> npt.NDArray:
        r"""
        Interleaves the input sequence $x[n]$.

        Arguments:
            x: The input sequence $x[n]$. Length must be a multiple of the interleaver size.

        Returns:
            The interleaved sequence $y[n]$.
        """
        x = verify_arraylike(x, ndim=1, size_multiple=len(self))

        y = np.empty_like(x)
        y.reshape((-1, len(self)))[..., self.map] = x.reshape((-1, len(self)))

        return convert_output(y)

    def deinterleave(self, y: npt.ArrayLike) -> npt.NDArray:
        r"""
        Deinterleaves the input sequence $y[n]$.

        Arguments:
            y: The input sequence $y[n]$. Length must be a multiple of the interleaver size.

        Returns:
            The deinterleaved sequence $x[n]$.
        """
        y = verify_arraylike(y, ndim=1, size_multiple=len(self))

        x = np.empty_like(y)
        x.reshape((-1, len(self)))[..., self.inverse_map] = y.reshape((-1, len(self)))

        return convert_output(x)

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
        verify_scalar(rows, int=True, positive=True)
        verify_scalar(cols, int=True, positive=True)

        map = np.arange(rows * cols).reshape((rows, cols)).T.ravel()
        super().__init__(map)
