"""
A module containing a class for phase-shift keying (PSK) modulation.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._helper import export


@export
class PSK:
    """
    Implements phase-shift keying (PSK) modulation and demodulation.

    Examples:
        See the :ref:`psk` example.

    Group:
        modulation
    """

    def __init__(self, order: int, offset: float = 0.0):
        r"""
        Creates a new PSK object.

        Arguments:
            order: The modulation order $M$. Must be at least 2.
            offset: The phase offset $\phi$ in radians.
        """
        if not order >= 2:
            raise ValueError("Argument 'order' must be at least 2")

        self._order = order
        self._offset = offset
        self._symbol_map = np.exp(1j * (2 * np.pi * np.arange(order) / order + offset))

    def modulate(self, symbols: npt.ArrayLike) -> np.ndarray:
        """
        Modulates to decimal symbols $s[k]$ to complex symbols $x[k]$.

        Arguments:
            symbols: The decimal symbols $s[k]$ to modulate, $0$ to $M-1$.

        Examples:
            See the :ref:`psk` example.
        """
        symbols = np.asarray(symbols)
        return self.symbol_map[symbols]

    def demodulate(self, x_hat: npt.ArrayLike) -> np.ndarray:
        r"""
        Demodulates the complex symbols $\hat{x}[k]$ to decimal symbols $\hat{s}[k]$
        using maximum-likelihood estimation.

        Arguments:
            x_hat: The complex symbols $\hat{x}[k]$ to demodulate.

        Returns:
            The decimal symbols $\hat{s}[k]$, $0$ to $M-1$.

        Examples:
            See the :ref:`psk` example.
        """
        x_hat = np.asarray(x_hat)
        error_vectors = np.subtract.outer(x_hat, self.symbol_map)
        s_hat = np.argmin(error_vectors, axis=-1)
        return s_hat

    @property
    def order(self) -> int:
        """
        The modulation order $M$.

        Examples:
            See the :ref:`psk` example.
        """
        return self._order

    @property
    def offset(self) -> float:
        r"""
        The phase offset $\phi$ in radians.

        Examples:
            See the :ref:`psk` example.
        """
        return self._offset

    @property
    def symbol_map(self) -> np.ndarray:
        r"""
        The symbol map ${0, \dots, M-1} -> \mathbb{C}$. This maps decimal symbols from $0$ to $M-1$
        to complex symbols.

        Examples:
            See the :ref:`psk` example.
        """
        return self._symbol_map
