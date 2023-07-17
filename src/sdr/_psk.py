"""
A module containing a class for phase-shift keying (PSK) modulation.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._helper import export
from ._probability import Q


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

    def symbol_error_rate(self, ebn0: npt.ArrayLike | None = None, esn0: npt.ArrayLike | None = None) -> np.ndarray:
        r"""
        Computes the symbol error rate (SER) at the provided SNRs.

        Arguments:
            ebn0: Bit energy $E_b$ to noise PSD $N_0$ ratio in dB. If `None`, `esn0` must be provided.
            esn0: Symbol energy $E_s$ to noise PSD $N_0$ ratio in dB. If `None`, `ebn0` must be provided.

        Returns:
            The symbol error rate $P_e$.

        References:
            - John Proakis, *Digital Communications*, Chapter 4: Optimum Receivers for AWGN Channels.
        """
        M = self.order  # Modulation order
        if ebn0 is not None:
            ebn0 = np.asarray(ebn0)
        elif esn0 is not None:
            esn0 = np.asarray(esn0)
            ebn0 = esn0 - 10 * np.log10(M)  # Bit energy to noise PSD ratio
        else:
            raise ValueError("Either 'ebn0' or 'esn0' must be provided.")

        ebn0_linear = 10 ** (ebn0 / 10)

        if M == 2:
            # Equation 4.3-13
            Pe = Q(np.sqrt(2 * ebn0_linear))
        elif M == 4:
            # Equation 4.3-15
            Pe = 2 * Q(np.sqrt(2 * ebn0_linear)) * (1 - 1 / 2 * Q(np.sqrt(2 * ebn0_linear)))
        else:
            # Equation 4.3-17
            # NOTE: This is an approximation for large M
            Pe = 2 * Q(np.sqrt(2 * np.log2(M) * np.sin(np.pi / M) ** 2 * ebn0_linear))

        return Pe

    def bit_error_rate(self, ebn0: npt.ArrayLike | None = None, esn0: npt.ArrayLike | None = None) -> np.ndarray:
        r"""
        Computes the bit error rate (BER) at the provided SNRs.

        Arguments:
            ebn0: Bit energy $E_b$ to noise PSD $N_0$ ratio in dB. If `None`, `esn0` must be provided.
            esn0: Symbol energy $E_s$ to noise PSD $N_0$ ratio in dB. If `None`, `ebn0` must be provided.

        Returns:
            The bit error rate $P_b$.

        References:
            - John Proakis, *Digital Communications*, Chapter 4: Optimum Receivers for AWGN Channels.
        """
        M = self.order  # Modulation order
        k = np.log2(M)  # Bits per symbol

        # Equation 4.3-20
        Pe = self.symbol_error_rate(ebn0, esn0) / k

        return Pe

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
