"""
A module containing classes for phase-shift keying (PSK) modulations.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.integrate
import scipy.special
from typing_extensions import Literal

from .._conversion import ebn0_to_esn0, esn0_to_ebn0, linear
from .._data import unpack
from .._helper import export, extend_docstring
from .._probability import Q
from ._linear import LinearModulation
from ._symbol_mapping import binary_code, gray_code


@export
class PSK(LinearModulation):
    r"""
    Implements phase-shift keying (PSK) modulation and demodulation.

    Notes:
        Phase-shift keying (PSK) is a linear phase modulation scheme that encodes information by modulating
        the phase of a carrier sinusoid. The modulation order $M = 2^k$ is a power of 2 and indicates the number of
        phases used. The input bit stream is taken $k$ bits at a time to create decimal symbols
        $s[k] \in \{0, \dots, M-1\}$. These decimal symbols $s[k]$ are then mapped to complex symbols
        $a[k] \in \mathbb{C}$ by the equation

        $$a[k] = \exp \left[ j\left(\frac{2\pi}{M}s[k] + \phi\right) \right] .$$

    Examples:
        Create a QPSK modem whose constellation has a 45° phase offset.

        .. ipython:: python

            qpsk = sdr.PSK(4, phase_offset=45, pulse_shape="srrc"); qpsk

            @savefig sdr_PSK_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.symbol_map(qpsk);

        Generate a random bit stream, convert to 2-bit symbols, and map to complex symbols.

        .. ipython:: python

            bits = np.random.randint(0, 2, 1000); bits[0:8]
            symbols = sdr.pack(bits, qpsk.bps); symbols[0:4]
            complex_symbols = qpsk.map_symbols(symbols); complex_symbols[0:4]

            @savefig sdr_PSK_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(complex_symbols, linestyle="-");

        Modulate and pulse shape the symbols to a complex baseband signal.

        .. ipython:: python

            tx_samples = qpsk.modulate(symbols)

            @savefig sdr_PSK_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(tx_samples[0:50*qpsk.sps], sample_rate=qpsk.sps);

            @savefig sdr_PSK_4.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.eye(tx_samples[5*qpsk.sps : -5*qpsk.sps], qpsk.sps);

        Add AWGN noise such that $E_b/N_0 = 20$ dB.

        .. ipython:: python

            ebn0 = 20; \
            snr = sdr.ebn0_to_snr(ebn0, bps=qpsk.bps, sps=qpsk.sps); \
            rx_samples = sdr.awgn(tx_samples, snr=snr)

            @savefig sdr_PSK_5.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(rx_samples[0:50*qpsk.sps], sample_rate=qpsk.sps);

        Matched filter and demodulate.

        .. ipython:: python

            rx_symbols, rx_complex_symbols = qpsk.demodulate(rx_samples)

            # The symbol decisions are error-free
            np.array_equal(symbols, rx_symbols)

            @savefig sdr_PSK_6.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(rx_complex_symbols);

        See the :ref:`psk` example.

    Group:
        modulation-linear
    """

    def __init__(
        self,
        order: int,
        phase_offset: float = 0.0,
        symbol_labels: Literal["bin", "gray"] | npt.ArrayLike = "gray",
        sps: int = 8,
        pulse_shape: npt.ArrayLike | Literal["rect", "rc", "srrc"] = "rect",
        span: int | None = None,
        alpha: float | None = None,
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

            sps: The number of samples per symbol $f_s / f_{sym}$.
            pulse_shape: The pulse shape $h[n]$ of the modulated signal.

                - `npt.ArrayLike`: A custom pulse shape. It is important that `sps` matches the design
                  of the pulse shape. See :ref:`pulse-shaping-functions`.
                - `"rect"`: Rectangular pulse shape.
                - `"rc"`: Raised cosine pulse shape.
                - `"srrc"`: Square-root raised cosine pulse shape.

            span: The span of the pulse shape in symbols. This is only used if `pulse_shape` is a string.
                If `None`, 1 is used for `"rect"` and 10 is used for `"rc"` and `"srrc"`.
            alpha: The roll-off factor of the pulse shape. If `None`, 0.2 is used for `"rc"` and `"srrc"`.

        See Also:
            sdr.rectangular, sdr.raised_cosine, sdr.root_raised_cosine
        """
        # Define the base PSK symbol map
        base_symbol_map = np.exp(1j * (2 * np.pi * np.arange(order) / order + np.deg2rad(phase_offset)))

        super().__init__(
            base_symbol_map,
            phase_offset=phase_offset,
            sps=sps,
            pulse_shape=pulse_shape,
            span=span,
            alpha=alpha,
        )

        if symbol_labels == "bin":
            self._symbol_labels = binary_code(self.bps)
            self._symbol_labels_str = "bin"
        elif symbol_labels == "gray":
            self._symbol_labels = gray_code(self.bps)
            self._symbol_labels_str = "gray"
        else:
            if not np.array_equal(np.sort(symbol_labels), np.arange(self.order)):
                raise ValueError(f"Argument 'symbol_labels' have unique values 0 to {self.order-1}.")
            self._symbol_labels = np.asarray(symbol_labels)
            self._symbol_labels_str = self._symbol_labels

        # Relabel the symbols
        self._symbol_map[self._symbol_labels] = self._symbol_map.copy()

    def __repr__(self) -> str:
        """
        Returns a code-styled string representation of the object.
        """
        return (
            f"sdr.{type(self).__name__}({self.order}, phase_offset={self.phase_offset}, "
            + f"symbol_labels={self._symbol_labels_str!r})"
        )

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the object.
        """
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  symbol_map: {self.symbol_map.shape} shape"
        string += f"\n    {self.symbol_map.tolist()}"
        string += f"\n  symbol_labels: {self._symbol_labels_str!r}"
        string += f"\n  phase_offset: {self.phase_offset}"
        return string

    def ber(self, ebn0: npt.ArrayLike | None = None, diff_encoded: bool = False) -> np.ndarray:
        r"""
        Computes the bit error rate (BER) at the provided $E_b/N_0$ values.

        Arguments:
            ebn0: Bit energy $E_b$ to noise PSD $N_0$ ratio in dB.
            diff_encoded: Indicates whether the input symbols were differentially encoded.

        Returns:
            The bit error rate $P_{be}$.

        See Also:
            sdr.esn0_to_ebn0, sdr.snr_to_ebn0

        References:
            - Simon and Alouini, *Digital Communications over Fading Channels*,
              Chapter 8: Performance of Single-Channel Receivers.
            - John Proakis, *Digital Communications*, Chapter 4: Optimum Receivers for AWGN Channels.

        Examples:
            See the :ref:`psk` example.

            Plot theoretical BER curves for BPSK, QPSK, 8-PSK, and 16-PSK in an AWGN channel.

            .. ipython:: python

                bpsk = sdr.PSK(2); \
                qpsk = sdr.PSK(4); \
                psk8 = sdr.PSK(8); \
                psk16 = sdr.PSK(16); \
                ebn0 = np.linspace(-2, 10, 100)

                @savefig sdr_psk_ber_1.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.ber(ebn0, bpsk.ber(ebn0), label="BPSK"); \
                sdr.plot.ber(ebn0, qpsk.ber(ebn0), label="QPSK"); \
                sdr.plot.ber(ebn0, psk8.ber(ebn0), label="8-PSK"); \
                sdr.plot.ber(ebn0, psk16.ber(ebn0), label="16-PSK"); \
                plt.title("BER curves for PSK modulation in an AWGN channel"); \
                plt.tight_layout();

            Compare the bit error rate of QPSK and DE-QPSK in an AWGN channel.

            .. ipython:: python

                @savefig sdr_psk_ber_2.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.ber(ebn0, qpsk.ber(ebn0), label="QPSK"); \
                sdr.plot.ber(ebn0, qpsk.ber(ebn0, diff_encoded=True), label="DE-QPSK"); \
                plt.title("BER curves for PSK and DE-PSK modulation in an AWGN channel"); \
                plt.tight_layout();
        """
        M = self.order
        k = self.bps
        ebn0 = np.asarray(ebn0)
        ebn0_linear = linear(ebn0)
        esn0 = ebn0_to_esn0(ebn0, k)
        esn0_linear = linear(esn0)

        if not diff_encoded:
            if M == 2:
                # Equation 4.3-13 from Proakis
                Pbe = Q(np.sqrt(2 * ebn0_linear))
            elif M == 4 and np.array_equal(self._symbol_labels, gray_code(k)):
                # Equation 4.3-13 from Proakis
                Pbe = Q(np.sqrt(2 * ebn0_linear))
            else:
                # Equation 8.29 from Simon and Alouini
                Pbe = np.zeros_like(esn0_linear)
                for i in range(esn0_linear.size):
                    for j in range(1, M):
                        Pj = Pk(M, esn0_linear[i], j)
                        # The number of bits that differ between symbol j and symbol 0
                        N_bits = unpack(self._symbol_labels[j] ^ self._symbol_labels[0], k).sum()
                        Pbe[i] += Pj * N_bits
                    # Equation 8.31 from Simon and Alouini
                    Pbe[i] /= k
        else:
            if M == 2:
                # Equation 8.37 from Simon and Alouini
                Pbe = 2 * Q(np.sqrt(2 * ebn0_linear)) - 2 * Q(np.sqrt(2 * ebn0_linear)) ** 2
            elif M == 4 and np.array_equal(self._symbol_labels, gray_code(k)):
                # Equation 8.37 from Simon and Alouini
                Pbe = 2 * Q(np.sqrt(2 * ebn0_linear)) - 2 * Q(np.sqrt(2 * ebn0_linear)) ** 2
            else:
                raise ValueError("Differential encoding is not supported for M-PSK with M > 4.")

        return Pbe

    def ser(self, esn0: npt.ArrayLike | None = None, diff_encoded: bool = False) -> np.ndarray:
        r"""
        Computes the symbol error rate (SER) at the provided $E_s/N_0$ values.

        Arguments:
            esn0: Symbol energy $E_s$ to noise PSD $N_0$ ratio in dB.
            diff_encoded: Indicates whether the input symbols were differentially encoded.

        Returns:
            The symbol error rate $P_{se}$.

        See Also:
            sdr.ebn0_to_esn0, sdr.snr_to_esn0

        References:
            - Simon and Alouini, *Digital Communications over Fading Channels*,
              Chapter 8: Performance of Single-Channel Receivers.
            - John Proakis, *Digital Communications*, Chapter 4: Optimum Receivers for AWGN Channels.

        Examples:
            See the :ref:`psk` example.

            Plot theoretical SER curves for BPSK, QPSK, 8-PSK, and 16-PSK in an AWGN channel.

            .. ipython:: python

                bpsk = sdr.PSK(2); \
                qpsk = sdr.PSK(4); \
                psk8 = sdr.PSK(8); \
                psk16 = sdr.PSK(16); \
                esn0 = np.linspace(-2, 10, 100)

                @savefig sdr_psk_ser_1.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.ser(esn0, bpsk.ser(esn0), label="BPSK"); \
                sdr.plot.ser(esn0, qpsk.ser(esn0), label="QPSK"); \
                sdr.plot.ser(esn0, psk8.ser(esn0), label="8-PSK"); \
                sdr.plot.ser(esn0, psk16.ser(esn0), label="16-PSK"); \
                plt.title("SER curves for PSK modulation in an AWGN channel"); \
                plt.tight_layout();

            Compare the symbol error rate of QPSK and DE-QPSK in an AWGN channel.

            .. ipython:: python

                @savefig sdr_psk_ser_2.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.ser(esn0, qpsk.ser(esn0), label="QPSK"); \
                sdr.plot.ser(esn0, qpsk.ser(esn0, diff_encoded=True), label="DE-QPSK"); \
                plt.title("SER curves for PSK and DE-PSK modulation in an AWGN channel"); \
                plt.tight_layout();
        """
        M = self.order
        k = self.bps
        esn0 = np.asarray(esn0)
        esn0_linear = linear(esn0)
        ebn0 = esn0_to_ebn0(esn0, k)
        ebn0_linear = linear(ebn0)

        if not diff_encoded:
            if M == 2:
                # Equation 4.3-13 from Proakis
                Pse = Q(np.sqrt(2 * ebn0_linear))
            elif M == 4:
                # Equation 4.3-15 from Proakis
                Pse = 2 * Q(np.sqrt(2 * ebn0_linear)) * (1 - 1 / 2 * Q(np.sqrt(2 * ebn0_linear)))
            else:
                # Equation 8.18 from Simon and Alouini
                Pse = np.zeros_like(esn0_linear)
                for i in range(esn0_linear.size):
                    Pse[i] = (
                        Q(np.sqrt(2 * esn0_linear[i]))
                        + scipy.integrate.quad(
                            lambda u, i: 2
                            / np.sqrt(np.pi)
                            * np.exp(-((u - np.sqrt(esn0_linear[i])) ** 2))
                            * Q(np.sqrt(2) * u * np.tan(np.pi / M)),
                            0,
                            np.inf,
                            args=(i,),
                        )[0]
                    )
        else:
            # Equation 8.36 from Simon and Alouini
            Pse_non_diff = self.ser(esn0, diff_encoded=False)
            Pse = 2 * Pse_non_diff - Pse_non_diff**2
            for i in range(esn0_linear.size):
                for j in range(1, M):
                    Pj = Pk(M, esn0_linear[i], j)
                    Pse[i] -= Pj**2

        return Pse

    @property
    @extend_docstring(
        LinearModulation.phase_offset,
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
        LinearModulation.symbol_map,
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


@export
class PiMPSK(PSK):
    r"""
    Implements $\pi/M$ phase-shift keying ($\pi/M$ PSK) modulation and demodulation.

    Notes:
        $\pi/M$ M-PSK is a linear phase modulation scheme similar to conventional M-PSK. One key distinction is that
        in $\pi/M$ M-PSK, the odd symbols are rotated by $\pi/M$ radians relative to the even symbols.
        This prevents symbol transitions through the origin, which results in a lower peak-to-average power ratio
        (PAPR).

        The modulation order $M = 2^k$ is a power of 2 and indicates the number of phases used $2M$.
        The input bit stream is taken $k$ bits at a time to create decimal symbols
        $s[k] \in \{0, \dots, M-1\}$. These decimal symbols $s[k]$ are then mapped to complex symbols
        $a[k] \in \mathbb{C}$ by the equation

        $$a[k] =
        \begin{cases}
        \displaystyle
        \exp \left[ j\left(\frac{2\pi}{M}s[k] + \phi\right) \right] & k\ \text{even} \\
        \displaystyle
        \exp \left[ j\left(\frac{2\pi}{M}s[k] + \phi + \pi/M\right) \right] & k\ \text{odd} \\
        \end{cases}
        $$

    Examples:
        Create a $\pi/4$ QPSK modem.

        .. ipython:: python

            pi4_qpsk = sdr.PiMPSK(4, pulse_shape="srrc"); pi4_qpsk

            @savefig sdr_PiMPSK_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.symbol_map(pi4_qpsk);

        Generate a random bit stream, convert to 2-bit symbols, and map to complex symbols.

        .. ipython:: python

            bits = np.random.randint(0, 2, 1000); bits[0:8]
            symbols = sdr.pack(bits, pi4_qpsk.bps); symbols[0:4]
            complex_symbols = pi4_qpsk.map_symbols(symbols); complex_symbols[0:4]

            @savefig sdr_PiMPSK_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(complex_symbols, linestyle="-");

        Modulate and pulse shape the symbols to a complex baseband signal.

        .. ipython:: python

            tx_samples = pi4_qpsk.modulate(symbols)

            @savefig sdr_PiMPSK_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(tx_samples[0:50*pi4_qpsk.sps], sample_rate=pi4_qpsk.sps);

            @savefig sdr_PiMPSK_4.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.eye(tx_samples[5*pi4_qpsk.sps : -5*pi4_qpsk.sps], pi4_qpsk.sps);

        Add AWGN noise such that $E_b/N_0 = 20$ dB.

        .. ipython:: python

            ebn0 = 20; \
            snr = sdr.ebn0_to_snr(ebn0, bps=pi4_qpsk.bps, sps=pi4_qpsk.sps); \
            rx_samples = sdr.awgn(tx_samples, snr=snr)

            @savefig sdr_PiMPSK_5.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(rx_samples[0:50*pi4_qpsk.sps], sample_rate=pi4_qpsk.sps);

        Matched filter and demodulate.

        .. ipython:: python

            rx_symbols, rx_complex_symbols = pi4_qpsk.demodulate(rx_samples)

            # The symbol decisions are error-free
            np.array_equal(symbols, rx_symbols)

            @savefig sdr_PiMPSK_6.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(rx_complex_symbols);

        See the :ref:`psk` example.

    Group:
        modulation-linear
    """

    def __init__(
        self,
        order: int,
        phase_offset: float = 0.0,
        symbol_labels: Literal["bin", "gray"] | npt.ArrayLike = "gray",
        sps: int = 8,
        pulse_shape: npt.ArrayLike | Literal["rect", "rc", "srrc"] = "rect",
        span: int | None = None,
        alpha: float | None = None,
    ):
        r"""
        Creates a new $\pi/M$ PSK object.

        Arguments:
            order: The modulation order $M = 2^k$, where $k \ge 1$ is the bits per symbol.
            phase_offset: The absolute phase offset $\phi$ in degrees.
            symbol_labels: The decimal symbol labels of consecutive complex symbols.

                - `"bin"`: The symbols are binary-coded. Adjacent symbols may differ by more than one bit.
                - `"gray":` The symbols are Gray-coded. Adjacent symbols only differ by one bit.
                - `npt.ArrayLike`: An $M$-length array whose indices are the default symbol labels and whose values are
                  the new symbol labels. The default symbol labels are $0$ to $M-1$ for phases starting at $1 + 0j$
                  and going counter-clockwise around the unit circle.

            sps: The number of samples per symbol $f_s / f_{sym}$.
            pulse_shape: The pulse shape $h[n]$ of the modulated signal.

                - `npt.ArrayLike`: A custom pulse shape. It is important that `sps` matches the design
                  of the pulse shape. See :ref:`pulse-shaping-functions`.
                - `"rect"`: Rectangular pulse shape.
                - `"rc"`: Raised cosine pulse shape.
                - `"srrc"`: Square-root raised cosine pulse shape.

            span: The span of the pulse shape in symbols. This is only used if `pulse_shape` is a string.
                If `None`, 1 is used for `"rect"` and 10 is used for `"rc"` and `"srrc"`.
            alpha: The roll-off factor of the pulse shape. If `None`, 0.2 is used for `"rc"` and `"srrc"`.

        See Also:
            sdr.rectangular, sdr.raised_cosine, sdr.root_raised_cosine
        """
        super().__init__(
            order,
            phase_offset=phase_offset,
            symbol_labels=symbol_labels,
            sps=sps,
            pulse_shape=pulse_shape,
        )

    def _map_symbols(self, s: npt.ArrayLike) -> np.ndarray:
        s = super()._map_symbols(s)

        # Rotate odd symbols by pi/M
        s_rotated = s.copy()
        s_rotated[1::2] *= np.exp(1j * np.pi / self.order)

        return s_rotated

    def _decide_symbols(self, a_hat: npt.ArrayLike) -> np.ndarray:
        a_hat = np.asarray(a_hat)

        # Rotate odd symbols by -pi/M
        a_hat_derotated = a_hat.copy()
        a_hat_derotated[1::2] *= np.exp(-1j * np.pi / self.order)

        return super()._decide_symbols(a_hat_derotated)


@export
class OQPSK(PSK):
    r"""
    Implements offset quadrature phase-shift keying (OQPSK) modulation and demodulation.

    Notes:
        Offset QPSK is a linear phase modulation scheme similar to conventional QPSK. One key distinction is that
        the I and Q channels transition independently, one half symbol apart. This prevents symbol transitions
        through the origin, which results in a lower peak-to-average power ratio (PAPR).

        The modulation order $M = 2^k$ is a power of 2 and indicates the number of phases used.
        The input bit stream is taken $k$ bits at a time to create decimal symbols
        $s[k] \in \{0, \dots, M-1\}$. These decimal symbols $s[k]$ are then mapped to complex symbols
        $a[k] \in \mathbb{C}$ by the equation

        $$I[k] + jQ[k] = \exp \left[ j\left(\frac{2\pi}{M}s[k] + \phi\right) \right]$$

        $$
        \begin{align}
        a[k + 0] &= I[k] + jQ[k - 1] \\
        a[k + 1/2] &= I[k] + jQ[k] \\
        a[k + 1] &= I[k + 1] + jQ[k] \\
        a[k + 3/2] &= I[k + 1] + jQ[k + 1] \\
        \end{align}
        $$

    Examples:
        Create a OQPSK modem.

        .. ipython:: python

            oqpsk = sdr.OQPSK(pulse_shape="srrc"); oqpsk

            @savefig sdr_OQPSK_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.symbol_map(oqpsk);

        Generate a random bit stream, convert to 2-bit symbols, and map to complex symbols.

        .. ipython:: python

            bits = np.random.randint(0, 2, 1000); bits[0:8]
            symbols = sdr.pack(bits, oqpsk.bps); symbols[0:4]
            complex_symbols = oqpsk.map_symbols(symbols); complex_symbols[0:4]

            @savefig sdr_OQPSK_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(complex_symbols, linestyle="-");

        Modulate and pulse shape the symbols to a complex baseband signal.

        .. ipython:: python

            tx_samples = oqpsk.modulate(symbols)

            @savefig sdr_OQPSK_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(tx_samples[0:50*oqpsk.sps], sample_rate=oqpsk.sps);

            @savefig sdr_OQPSK_4.png
            plt.figure(figsize=(8, 6)); \
            plt.subplot(2, 1, 1); \
            sdr.plot.eye(tx_samples[5*oqpsk.sps : -5*oqpsk.sps].real, oqpsk.sps); \
            plt.title("In-phase channel, $I$"); \
            plt.subplot(2, 1, 2); \
            sdr.plot.eye(tx_samples[5*oqpsk.sps : -5*oqpsk.sps].imag, oqpsk.sps); \
            plt.title("Quadrature channel, $Q$"); \
            plt.tight_layout();

        Add AWGN noise such that $E_b/N_0 = 20$ dB.

        .. ipython:: python

            ebn0 = 20; \
            snr = sdr.ebn0_to_snr(ebn0, bps=oqpsk.bps, sps=oqpsk.sps); \
            rx_samples = sdr.awgn(tx_samples, snr=snr)

            @savefig sdr_OQPSK_5.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(rx_samples[0:50*oqpsk.sps], sample_rate=oqpsk.sps);

        Matched filter and demodulate. Note, the first symbol has $Q = 0$ and the last symbol has $I = 0$.

        .. ipython:: python

            rx_symbols, rx_complex_symbols = oqpsk.demodulate(rx_samples)

            # The symbol decisions are error-free
            np.array_equal(symbols, rx_symbols)

            @savefig sdr_OQPSK_6.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(rx_complex_symbols);

        See the :ref:`psk` example.

    Group:
        modulation-linear
    """

    def __init__(
        self,
        phase_offset: float = 45,
        symbol_labels: Literal["bin", "gray"] | npt.ArrayLike = "gray",
        sps: int = 8,
        pulse_shape: npt.ArrayLike | Literal["rect", "rc", "srrc"] = "rect",
        span: int | None = None,
        alpha: float | None = None,
    ):
        r"""
        Creates a new OQPSK object.

        Arguments:
            phase_offset: The absolute phase offset $\phi$ in degrees.
            symbol_labels: The decimal symbol labels of consecutive complex symbols.

                - `"bin"`: The symbols are binary-coded. Adjacent symbols may differ by more than one bit.
                - `"gray":` The symbols are Gray-coded. Adjacent symbols only differ by one bit.
                - `npt.ArrayLike`: An $4$-length array whose indices are the default symbol labels and whose values are
                  the new symbol labels. The default symbol labels are $0$ to $4-1$ for phases starting at $1 + 0j$
                  and going counter-clockwise around the unit circle.

            sps: The number of samples per symbol $f_s / f_{sym}$.
            pulse_shape: The pulse shape $h[n]$ of the modulated signal.

                - `npt.ArrayLike`: A custom pulse shape. It is important that `sps` matches the design
                  of the pulse shape. See :ref:`pulse-shaping-functions`.
                - `"rect"`: Rectangular pulse shape.
                - `"rc"`: Raised cosine pulse shape.
                - `"srrc"`: Square-root raised cosine pulse shape.

            span: The span of the pulse shape in symbols. This is only used if `pulse_shape` is a string.
                If `None`, 1 is used for `"rect"` and 10 is used for `"rc"` and `"srrc"`.
            alpha: The roll-off factor of the pulse shape. If `None`, 0.2 is used for `"rc"` and `"srrc"`.

        See Also:
            sdr.rectangular, sdr.raised_cosine, sdr.root_raised_cosine
        """
        super().__init__(
            4,
            phase_offset=phase_offset,
            symbol_labels=symbol_labels,
            sps=sps,
            pulse_shape=pulse_shape,
            span=span,
            alpha=alpha,
        )

        if sps > 1 and sps % 2 != 0:
            raise ValueError(f"Argument 'sps' must be even, not {sps}.")

    def __repr__(self) -> str:
        """
        Returns a code-styled string representation of the object.
        """
        return f"sdr.{type(self).__name__}(phase_offset={self.phase_offset}, symbol_labels={self._symbol_labels_str!r})"

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the object.
        """
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  symbol_map: {self.symbol_map.shape} shape"
        string += f"\n    {self.symbol_map.tolist()}"
        string += f"\n  symbol_labels: {self._symbol_labels_str!r}"
        string += f"\n  phase_offset: {self.phase_offset}"
        return string

    def _map_symbols(self, s: npt.ArrayLike) -> np.ndarray:
        a = super()._map_symbols(s)

        a_I = np.repeat(a.real, 2)
        a_Q = np.repeat(a.imag, 2)

        # Shift Q symbols by 1/2 symbol
        a_I = np.append(a_I, 0)
        a_Q = np.insert(a_Q, 0, 0)

        a = a_I + 1j * a_Q

        return a

    def _tx_pulse_shape(self, a: npt.ArrayLike) -> np.ndarray:
        a = np.asarray(a)  # Complex symbols
        a_I, a_Q = a.real, a.imag

        # Shift Q symbols by -1/2 symbol and grab 1 sample per symbol
        a_I = a_I[:-1:2]
        a_Q = a_Q[1::2]

        x_I = self._tx_filter(a_I, mode="full")  # Complex samples
        x_Q = self._tx_filter(a_Q, mode="full")  # Complex samples

        # Shift Q symbols by 1/2 symbol
        x_I = np.append(x_I, np.zeros(self.sps // 2))
        x_Q = np.insert(x_Q, 0, np.zeros(self.sps // 2))

        x = x_I + 1j * x_Q

        return x

    def _decide_symbols(self, a_hat: npt.ArrayLike) -> np.ndarray:
        a_hat = np.asarray(a_hat)
        a_hat_I, a_hat_Q = a_hat.real, a_hat.imag

        # Shift Q symbols by -1/2 symbol and grab 1 sample per symbol
        a_hat_I = a_hat_I[:-1:2]
        a_hat_Q = a_hat_Q[1::2]

        a_hat = a_hat_I + 1j * a_hat_Q

        return super()._decide_symbols(a_hat)

    def _rx_matched_filter(self, x_hat: npt.ArrayLike) -> np.ndarray:
        x_hat = np.asarray(x_hat)
        x_hat_I, x_hat_Q = x_hat.real, x_hat.imag

        # Shift Q samples by -1/2 symbol
        x_hat_I = x_hat_I[: -self.sps // 2]
        x_hat_Q = x_hat_Q[self.sps // 2 :]

        a_hat_I = super()._rx_matched_filter(x_hat_I)  # Complex samples
        a_hat_Q = super()._rx_matched_filter(x_hat_Q)  # Complex samples

        a_hat_I = np.repeat(a_hat_I, 2)
        a_hat_Q = np.repeat(a_hat_Q, 2)

        # Shift Q symbols by 1/2 symbol
        a_hat_I = np.append(a_hat_I, 0)
        a_hat_Q = np.insert(a_hat_Q, 0, 0)

        a_hat = a_hat_I + 1j * a_hat_Q

        return a_hat


def Pk(M: int, esn0_linear: float, j: int) -> float:
    """
    Determines the probability of receiving symbol j given symbol 0 was transmitted.
    """
    # Equation 8.30 from Simon and Alouini
    A = scipy.integrate.quad(
        lambda theta, j: 1
        / (2 * np.pi)
        * np.exp(-esn0_linear * np.sin((2 * j - 1) * np.pi / M) ** 2 / np.sin(theta) ** 2),
        0,
        np.pi * (1 - (2 * j - 1) / M),
        args=(j,),
    )[0]
    B = scipy.integrate.quad(
        lambda theta, j: 1
        / (2 * np.pi)
        * np.exp(-esn0_linear * np.sin((2 * j + 1) * np.pi / M) ** 2 / np.sin(theta) ** 2),
        0,
        np.pi * (1 - (2 * j + 1) / M),
        args=(j,),
    )[0]

    # Probability of landing in decision region for symbol j when symbol 0 was transmitted
    return A - B
