"""
A module containing a class for phase-shift keying (PSK) modulation.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._helper import export, extend_docstring
from .._probability import Q
from .._snr import ebn0_to_esn0, esn0_to_ebn0
from ._linear import _LinearModulation
from ._symbol_mapping import binary_code, gray_code


@export
class PSK(_LinearModulation):
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
        phase_offset: float = 0.0,
        symbol_labels: Literal["bin", "gray"] | npt.ArrayLike = "gray",
    ):
        r"""
        Creates a new PSK object.

        Arguments:
            order: The modulation order $M = 2^k$, where $k \ge 1$ is the bits per symbol.
            phase_offset: The phase offset $\phi$ in degrees.
            symbol_labels: The decimal symbol labels of consecutive complex symbols.

                - `"bin"`: The symbols are binary-coded. Adjacent symbols may differ by more than one bit.
                - `"gray":` The symbols are Gray-coded. Adjacent symbols only differ by one bit.
                - `npt.ArrayLike`: An $M$-length array whose indices are the default symbol labels and whose values are
                  the new symbol labels. The default symbol labels are $0$ to $M-1$ for phases starting at $1 + 0j$
                  and going counter-clockwise around the unit circle.
        """
        # Define the base PSK symbol map
        base_symbol_map = np.exp(1j * (2 * np.pi * np.arange(order) / order + np.deg2rad(phase_offset)))

        super().__init__(order, base_symbol_map, phase_offset)

        if symbol_labels == "bin":
            self._symbol_labels = binary_code(self.bps)
        elif symbol_labels == "gray":
            self._symbol_labels = gray_code(self.bps)
        else:
            if not np.array_equal(np.sort(symbol_labels), np.arange(self.order)):
                raise ValueError(f"Argument 'symbol_labels' have unique values 0 to {self.order-1}.")
            self._symbol_labels = np.asarray(symbol_labels)

        # Relabel the symbols
        self._symbol_map[self._symbol_labels] = self._symbol_map.copy()

    @extend_docstring(
        _LinearModulation.bit_error_rate,
        {},
        r"""
        References:
            - John Proakis, *Digital Communications*, Chapter 4: Optimum Receivers for AWGN Channels.

        Examples:
            See the :ref:`psk` example.

            Plot theoretical BER curves for BPSK, QPSK, and 8-PSK in an AWGN channel.

            .. ipython:: python

                bpsk = sdr.PSK(2); \
                qpsk = sdr.PSK(4); \
                psk8 = sdr.PSK(8); \
                ebn0 = np.linspace(0, 10, 1000)

                @savefig sdr_psk_bit_error_rate_1.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.ber(ebn0, bpsk.bit_error_rate(ebn0), label="BPSK"); \
                sdr.plot.ber(ebn0, qpsk.bit_error_rate(ebn0), label="QPSK"); \
                sdr.plot.ber(ebn0, psk8.bit_error_rate(ebn0), label="8-PSK"); \
                plt.title("BER curves for BPSK, QPSK, and 8-PSK in an AWGN channel"); \
                plt.tight_layout();
        """,
    )
    def bit_error_rate(self, ebn0: npt.ArrayLike | None = None) -> np.ndarray:
        M = self.order
        k = self.bps
        ebn0 = np.asarray(ebn0)
        ebn0_linear = 10 ** (ebn0 / 10)

        if M in [2, 4]:
            # Equation 4.3-13
            Pe = Q(np.sqrt(2 * ebn0_linear))
        else:
            # Equation 4.3-20
            esn0 = ebn0_to_esn0(ebn0, k)
            Pe = self.symbol_error_rate(esn0) / k

        return Pe

    @extend_docstring(
        _LinearModulation.symbol_error_rate,
        {},
        r"""
        References:
            - John Proakis, *Digital Communications*, Chapter 4: Optimum Receivers for AWGN Channels.

        Examples:
            See the :ref:`psk` example.

            Plot theoretical SER curves for BPSK, QPSK, and 8-PSK in an AWGN channel.

            .. ipython:: python

                bpsk = sdr.PSK(2); \
                qpsk = sdr.PSK(4); \
                psk8 = sdr.PSK(8); \
                esn0 = np.linspace(0, 10, 1000)

                @savefig sdr_psk_symbol_error_rate_1.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.ser(esn0, bpsk.symbol_error_rate(esn0), label="BPSK"); \
                sdr.plot.ser(esn0, qpsk.symbol_error_rate(esn0), label="QPSK"); \
                sdr.plot.ser(esn0, psk8.symbol_error_rate(esn0), label="8-PSK"); \
                plt.title("SER curves for BPSK, QPSK, and 8-PSK in an AWGN channel"); \
                plt.tight_layout();
        """,
    )
    def symbol_error_rate(self, esn0: npt.ArrayLike | None = None) -> np.ndarray:
        M = self.order
        k = self.bps
        esn0 = np.asarray(esn0)
        # esn0_linear = 10 ** (esn0 / 10)
        ebn0 = esn0_to_ebn0(esn0, k)
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
    @extend_docstring(
        _LinearModulation.phase_offset,
        {},
        r"""
        Examples:
            See the :ref:`psk` example.

            Create a QPSK constellation with no phase offset.

            .. ipython:: python

                psk = sdr.PSK(4); \
                psk.phase_offset

                @savefig sdr_psk_phase_offset_1.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.symbol_map(psk.symbol_map);

            Create a QPSK constellation with 45° phase offset.

            .. ipython:: python

                psk = sdr.PSK(4, phase_offset=45); \
                psk.phase_offset

                @savefig sdr_psk_phase_offset_2.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.symbol_map(psk.symbol_map);
        """,
    )
    def phase_offset(self) -> float:
        return super().phase_offset

    @property
    @extend_docstring(
        _LinearModulation.symbol_map,
        {},
        r"""
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
        """,
    )
    def symbol_map(self) -> np.ndarray:
        return super().symbol_map
