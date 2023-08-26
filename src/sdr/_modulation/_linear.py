"""
A module containing a base class for linear phase/amplitude modulations.
"""
from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt

from .._helper import export


@export
class LinearModulation:
    """
    Implements linear phase/amplitude modulation with arbitrary symbol mapping.

    Group:
        modulation-linear
    """

    def __init__(
        self,
        symbol_map: npt.ArrayLike,
        phase_offset: float = 0.0,
    ):
        r"""
        Creates a new linear phase/amplitude modulation object.

        Arguments:
            symbol_map: The symbol mapping $\{0, \dots, M-1\} \mapsto \mathbb{C}$. An $M$-length array whose indices
                are decimal symbols and whose values are complex symbols, where $M$ is the modulation order.
            phase_offset: The phase offset $\phi$ in degrees.
        """
        symbol_map = np.asarray(symbol_map)
        if not symbol_map.ndim == 1:
            raise ValueError(f"Argument 'symbol_map' must be 1-D, not {symbol_map.ndim}-D.")
        if not np.log2(symbol_map.size).is_integer():
            raise ValueError(f"Argument 'symbol_map' must have a size that is a power of 2, not {symbol_map.size}.")
        self._symbol_map = symbol_map  # Decimal-to-complex symbol map
        self._order = symbol_map.size  # Modulation order
        self._bps = int(np.log2(self._order))  # Bits per symbol

        if not isinstance(phase_offset, (int, float)):
            raise TypeError(f"Argument 'phase_offset' must be a number, not {type(phase_offset)}.")
        self._phase_offset = phase_offset  # Phase offset in degrees

    def modulate(self, symbols: npt.ArrayLike) -> np.ndarray:
        r"""
        Modulates to decimal symbols $s[k]$ to complex symbols $x[k]$.

        Arguments:
            symbols: The decimal symbols $s[k]$ to modulate, $0$ to $M-1$.
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
        """
        x_hat = np.asarray(x_hat)
        error_vectors = np.subtract.outer(x_hat, self.symbol_map)
        s_hat = np.argmin(np.abs(error_vectors), axis=-1)
        return s_hat

    @abc.abstractmethod
    def ber(self, ebn0: npt.ArrayLike | None = None) -> np.ndarray:
        r"""
        Computes the bit error rate (BER) at the provided $E_b/N_0$ values.

        Arguments:
            ebn0: Bit energy $E_b$ to noise PSD $N_0$ ratio in dB.

        Returns:
            The bit error rate $P_b$.

        See Also:
            sdr.esn0_to_ebn0, sdr.snr_to_ebn0
        """
        raise NotImplementedError("Bit error rate calculation for arbitrary linear modulations is not supported.")

    @abc.abstractmethod
    def ser(self, esn0: npt.ArrayLike) -> np.ndarray:
        r"""
        Computes the symbol error rate (SER) at the provided $E_s/N_0$ values.

        Arguments:
            esn0: Symbol energy $E_s$ to noise PSD $N_0$ ratio in dB.

        Returns:
            The symbol error rate $P_e$.

        See Also:
            sdr.ebn0_to_esn0, sdr.snr_to_esn0
        """
        raise NotImplementedError("Symbol error rate calculation for arbitrary linear modulations is not supported.")

    @property
    def order(self) -> int:
        r"""
        The modulation order $M = 2^k$.
        """
        return self._order

    @property
    def bps(self) -> int:
        r"""
        The number of bits per symbol $k = \log_2 M$.
        """
        return self._bps

    @property
    def phase_offset(self) -> float:
        r"""
        The phase offset $\phi$ in degrees.
        """
        return self._phase_offset

    @property
    def symbol_map(self) -> np.ndarray:
        r"""
        The symbol map $\{0, \dots, M-1\} \mapsto \mathbb{C}$. This maps decimal symbols from $0$ to $M-1$
        to complex symbols.
        """
        return self._symbol_map
