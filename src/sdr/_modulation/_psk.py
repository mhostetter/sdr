"""
A module containing a class for phase-shift keying (PSK) modulation.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._helper import export
from .._probability import Q
from ._symbol_mapping import binary_code, gray_code


@export
class PSK:
    """
    Implements phase-shift keying (PSK) modulation and demodulation.

    Examples:
        See the :ref:`psk` example.

    Group:
        modulation-classes
    """

    def __init__(
        self,
        order: int,
        offset: float = 0.0,
        symbol_labels: Literal["bin", "gray"] | npt.ArrayLike = "gray",
    ):
        r"""
        Creates a new PSK object.

        Arguments:
            order: The modulation order $M = 2^k$ with $k \ge 1$.
            offset: The phase offset $\phi$ in degrees.
            symbol_labels: The symbol labels of consecutive symbols. If `"bin"`, the symbols are binary-coded.
                If `"gray"`, the symbols are Gray-coded. If an array-like object, the symbols are labeled by the
                values in the array. The array must have unique values from $0$ to $M-1$.
        """
        if not isinstance(order, int):
            raise TypeError(f"Argument 'order' must be an integer, not {type(order)}.")
        if not np.log2(order).is_integer():
            raise ValueError(f"Argument 'order' must be a power of 2, not {order}.")
        self._order = order
        self._k = int(np.log2(order))
        self._offset = offset

        if symbol_labels == "bin":
            self._symbol_labels = binary_code(self._k)
        elif symbol_labels == "gray":
            self._symbol_labels = gray_code(self._k)
        else:
            if not np.array_equal(np.sort(symbol_labels), np.arange(order)):
                raise ValueError(f"Argument 'symbol_labels' have unique values 0 to {order-1}.")
            self._symbol_labels = np.asarray(symbol_labels)

        # Define the base binary symbol map
        self._symbol_map = np.exp(1j * (2 * np.pi * np.arange(order) / order + np.deg2rad(offset)))

        # Relabel the symbols based on the symbol map
        self._symbol_map[self._symbol_labels] = self._symbol_map.copy()

    def modulate(self, symbols: npt.ArrayLike) -> np.ndarray:
        r"""
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
        s_hat = np.argmin(np.abs(error_vectors), axis=-1)
        return s_hat

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
        if ebn0 is not None:
            ebn0 = np.asarray(ebn0)
        elif esn0 is not None:
            esn0 = np.asarray(esn0)
            ebn0 = esn0 - 10 * np.log10(M)  # Bit energy to noise PSD ratio
        else:
            raise ValueError("Argument 'ebn0' or 'esn0' must be provided.")

        ebn0_linear = 10 ** (ebn0 / 10)

        if M in [2, 4]:
            # Equation 4.3-13
            Pe = Q(np.sqrt(2 * ebn0_linear))
        else:
            # Equation 4.3-20
            k = np.log2(M)  # Bits per symbol
            Pe = self.symbol_error_rate(ebn0, esn0) / k

        return Pe

    def symbol_error_rate(self, ebn0: npt.ArrayLike | None = None, esn0: npt.ArrayLike | None = None) -> np.ndarray:
        r"""
        Computes the symbol error rate (SER) at the provided SNRs.

        Arguments:
            esn0: Symbol energy $E_s$ to noise PSD $N_0$ ratio in dB. If `None`, `ebn0` must be provided.
            ebn0: Bit energy $E_b$ to noise PSD $N_0$ ratio in dB. If `None`, `esn0` must be provided.

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
            raise ValueError("Argument 'ebn0' or 'esn0' must be provided.")

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

    @property
    def order(self) -> int:
        r"""
        The modulation order $M = 2^k$.

        Examples:
            See the :ref:`psk` example.

            .. ipython:: python

                psk = sdr.PSK(4); \
                psk.order

                @savefig sdr_psk_order_1.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.symbol_map(psk.symbol_map);

            .. ipython:: python

                psk = sdr.PSK(8); \
                psk.order

                @savefig sdr_psk_offset_2.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.symbol_map(psk.symbol_map);
        """
        return self._order

    @property
    def offset(self) -> float:
        r"""
        The phase offset $\phi$ in degrees.

        Examples:
            See the :ref:`psk` example.

            .. ipython:: python

                psk = sdr.PSK(4); \
                psk.offset

                @savefig sdr_psk_offset_1.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.symbol_map(psk.symbol_map);

            .. ipython:: python

                psk = sdr.PSK(4, offset=45); \
                psk.offset

                @savefig sdr_psk_offset_2.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.symbol_map(psk.symbol_map);
        """
        return self._offset

    @property
    def symbol_labels(self) -> np.ndarray:
        r"""
        The symbols values (labels) of consecutive symbols.

        Examples:
            The default Gray-coded symbols. Adjacent symbols only differ by one bit.

            .. ipython:: python

                psk = sdr.PSK(8); \
                psk.symbol_labels

                @savefig sdr_psk_symbol_labels_1.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.symbol_map(psk.symbol_map, annotate="bin");

            The binary-coded symbols. Adjacent symbols may differ by more than one bit.

            .. ipython:: python

                psk = sdr.PSK(8, symbol_labels="bin"); \
                psk.symbol_labels

                @savefig sdr_psk_symbol_labels_2.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.symbol_map(psk.symbol_map, annotate="bin");
        """
        return self._symbol_labels

    @property
    def symbol_map(self) -> np.ndarray:
        r"""
        The symbol map $\{0, \dots, M-1\} \mapsto \mathbb{C}$. This maps decimal symbols from $0$ to $M-1$
        to complex symbols.

        Examples:
            See the :ref:`psk` example.

            The default Gray-coded symbols. Adjacent symbols only differ by one bit.

            .. ipython:: python

                psk = sdr.PSK(8); \
                psk.symbol_map

                @savefig sdr_psk_symbol_map_1.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.symbol_map(psk.symbol_map, annotate="bin");

            The binary-coded symbols. Adjacent symbols may differ by more than one bit.

            .. ipython:: python

                psk = sdr.PSK(8, symbol_labels="bin"); \
                psk.symbol_map

                @savefig sdr_psk_symbol_map_2.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.symbol_map(psk.symbol_map, annotate="bin");
        """
        return self._symbol_map
