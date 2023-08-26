"""
A module containing a class for phase-shift keying (PSK) modulations.
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
        $x[k] \in \mathbb{C}$ by the equation

        $$x[k] = \exp \left[ j\left(\frac{2\pi}{M}s[k] + \phi\right) \right] .$$

    Examples:
        Create a QPSK constellation with 45° phase offset.

        .. ipython:: python

            psk = sdr.PSK(4, phase_offset=45); psk

            @savefig sdr_PSK_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.symbol_map(psk);

        Generate a random bit stream, convert to 2-bit symbols, and modulate.

        .. ipython:: python

            bits = np.random.randint(0, 2, 1000); bits[0:8]
            symbols = sdr.pack(bits, psk.bps); symbols[0:4]
            a = psk.modulate(symbols); a[0:4]

            @savefig sdr_PSK_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(a, linestyle="-");

        Apply square-root raised-cosine pulse shaping.

        .. ipython:: python

            sps = 8; \
            h_rrc = sdr.root_raised_cosine(0.5, 10, sps); \
            tx_fir = sdr.Interpolator(sps, h_rrc); \
            x = tx_fir(a)

            @savefig sdr_PSK_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x[0:30*sps], sample_rate=sps);

        Add AWGN noise such that $E_b/N_0 = 12$ dB.

        .. ipython:: python

            ebn0 = 12; \
            snr = sdr.ebn0_to_snr(ebn0, bps=psk.bps, sps=sps); \
            y = sdr.awgn(x, snr=snr)

            @savefig sdr_PSK_4.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(y[0:30*sps], sample_rate=sps);

        Apply matched filtering.

        .. ipython:: python

            rx_fir = sdr.Decimator(sps, h_rrc); \
            y_hat = rx_fir(y)

            @savefig sdr_PSK_5.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(y_hat);

        See the :ref:`psk` example.

    Group:
        modulation-linear
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

        super().__init__(base_symbol_map, phase_offset)

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
    Implements $\pi/M$ M-PSK modulation and demodulation.

    Notes:
        $\pi/M$ M-PSK is a linear phase modulation scheme similar to conventional M-PSK. One key distinction is that
        in $\pi/M$ M-PSK, the odd symbols are rotated by $\pi/M$ radians relative to the even symbols.
        This prevents symbol transitions through the origin, which results in a lower peak-to-average power ratio
        (PAPR).

        The modulation order $M = 2^k$ is a power of 2 and indicates the number of phases used $2M$.
        The input bit stream is taken $k$ bits at a time to create decimal symbols
        $s[k] \in \{0, \dots, M-1\}$. These decimal symbols $s[k]$ are then mapped to complex symbols
        $x[k] \in \mathbb{C}$ by the equation

        $$x[k] =
        \begin{cases}
        \displaystyle
        \exp \left[ j\left(\frac{2\pi}{M}s[k] + \phi\right) \right] & k\ \text{even} \\
        \displaystyle
        \exp \left[ j\left(\frac{2\pi}{M}s[k] + \phi + \pi/M\right) \right] & k\ \text{odd} \\
        \end{cases}
        $$

    Examples:
        Create a $\pi/4$ QPSK constellation.

        .. ipython:: python

            psk = sdr.PiMPSK(4); psk

            @savefig sdr_PiMPSK_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.symbol_map(psk);

        Generate a random bit stream, convert to 2-bit symbols, and modulate.

        .. ipython:: python

            bits = np.random.randint(0, 2, 1000); bits[0:8]
            symbols = sdr.pack(bits, psk.bps); symbols[0:4]
            a = psk.modulate(symbols); a[0:4]

            @savefig sdr_PiMPSK_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(a, linestyle="-");

        Apply square-root raised-cosine pulse shaping.

        .. ipython:: python

            sps = 8; \
            h_rrc = sdr.root_raised_cosine(0.5, 10, sps); \
            tx_fir = sdr.Interpolator(sps, h_rrc); \
            x = tx_fir(a)

            @savefig sdr_PiMPSK_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x[0:30*sps], sample_rate=sps);

        Add AWGN noise such that $E_b/N_0 = 12$ dB.

        .. ipython:: python

            ebn0 = 12; \
            snr = sdr.ebn0_to_snr(ebn0, bps=psk.bps, sps=sps); \
            y = sdr.awgn(x, snr=snr)

            @savefig sdr_PiMPSK_4.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(y[0:30*sps], sample_rate=sps);

        Apply matched filtering.

        .. ipython:: python

            rx_fir = sdr.Decimator(sps, h_rrc); \
            y_hat = rx_fir(y)

            @savefig sdr_PiMPSK_5.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.constellation(y_hat);

        See the :ref:`psk` example.

    Group:
        modulation-linear
    """

    def __init__(
        self,
        order: int,
        phase_offset: float = 0.0,
        symbol_labels: Literal["bin", "gray"] | npt.ArrayLike = "gray",
    ):
        r"""
        Creates a new $\pi/M$ M-PSK object.

        Arguments:
            order: The modulation order $M = 2^k$, where $k \ge 1$ is the bits per symbol.
            phase_offset: The absolute phase offset $\phi$ in degrees.
            symbol_labels: The decimal symbol labels of consecutive complex symbols.

                - `"bin"`: The symbols are binary-coded. Adjacent symbols may differ by more than one bit.
                - `"gray":` The symbols are Gray-coded. Adjacent symbols only differ by one bit.
                - `npt.ArrayLike`: An $M$-length array whose indices are the default symbol labels and whose values are
                  the new symbol labels. The default symbol labels are $0$ to $M-1$ for phases starting at $1 + 0j$
                  and going counter-clockwise around the unit circle.
        """
        super().__init__(order, phase_offset=phase_offset, symbol_labels=symbol_labels)

    def modulate(self, symbols: npt.ArrayLike) -> np.ndarray:
        symbols = super().modulate(symbols)

        # Rotate odd symbols by pi/M
        symbols[1::2] *= np.exp(1j * np.pi / self.order)

        return symbols

    def demodulate(self, x_hat: npt.ArrayLike) -> np.ndarray:
        x_hat = np.asarray(x_hat)

        # Rotate odd symbols by -pi/M
        x_hat[1::2] *= np.exp(-1j * np.pi / self.order)

        return super().demodulate(x_hat)


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
