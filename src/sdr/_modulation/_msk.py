"""
A module containing classes for minimum-shift keying (MSK) modulations.
"""

from __future__ import annotations

import numpy.typing as npt
from typing_extensions import Literal

from .._helper import export, verify_scalar
from ._psk import OQPSK
from ._pulse_shapes import half_sine


@export
class MSK(OQPSK):
    r"""
    Implements minimum-shift keying (MSK) modulation and demodulation.

    Notes:
        MSK is a linear phase modulation scheme similar to OQPSK. One key distinction is that the pulse
        shape is a half sine wave. This results in a constant envelope signal, which results in a lower
        peak-to-average power ratio (PAPR).

        MSK can also be consider as continuous-phase frequency-shift keying (CPFSK) with the frequency separation
        equaling half the bit period.

    .. nomenclature::
        :collapsible:

        - $k$: Symbol index
        - $n$: Sample index
        - $s[k]$: Decimal symbols
        - $a[k]$ Complex symbols
        - $x[n]$: Pulse-shaped complex samples
        - $\tilde{x}[n]$: Received (noisy) pulse-shaped complex samples
        - $\tilde{a}[k]$: Received (noisy) complex symbols
        - $\hat{a}[k]$: Complex symbol decisions
        - $\hat{s}[k]$: Decimal symbol decisions

    Examples:
        Create a MSK modem.

        .. ipython:: python

            msk = sdr.MSK(); msk

            @savefig sdr_MSK_1.svg
            plt.figure(); \
            sdr.plot.symbol_map(msk);

        Generate a random bit stream, convert to 2-bit symbols, and map to complex symbols.

        .. ipython:: python

            bits = np.random.randint(0, 2, 1000); bits[0:8]
            symbols = sdr.pack(bits, msk.bps); symbols[0:4]
            complex_symbols = msk.map_symbols(symbols); complex_symbols[0:4]

            @savefig sdr_MSK_2.svg
            plt.figure(); \
            sdr.plot.constellation(complex_symbols, linestyle="-");

        Modulate and pulse shape the symbols to a complex baseband signal.

        .. ipython:: python

            tx_samples = msk.modulate(symbols)

            @savefig sdr_MSK_3.svg
            plt.figure(); \
            sdr.plot.time_domain(tx_samples[0:50*msk.sps]);

        MSK, like OQPSK, has I and Q channels that are offset by half a symbol period.

        .. ipython:: python

            @savefig sdr_MSK_4.svg
            plt.figure(figsize=(8, 6)); \
            sdr.plot.eye(tx_samples[5*msk.sps : -5*msk.sps], msk.sps); \
            plt.suptitle("Noiseless transmitted signal");

        The phase trajectory of MSK is linear and continuous. Although, it should be noted that the phase is not
        differentiable at the symbol boundaries. This leads to lower spectral efficiency than, for instance,
        GMSK.

        .. ipython:: python

            @savefig sdr_MSK_5.svg
            plt.figure(); \
            sdr.plot.phase_tree(tx_samples[msk.sps:], msk.sps);

        Add AWGN noise such that $E_b/N_0 = 30$ dB.

        .. ipython:: python

            ebn0 = 30; \
            snr = sdr.ebn0_to_snr(ebn0, bps=msk.bps, sps=msk.sps); \
            rx_samples = sdr.awgn(tx_samples, snr=snr)

            @savefig sdr_MSK_6.svg
            plt.figure(); \
            sdr.plot.time_domain(rx_samples[0:50*msk.sps]);

        Manually apply a matched filter. Examine the eye diagram of the matched filtered received signal.

        .. ipython:: python

            mf = sdr.FIR(msk.pulse_shape); \
            mf_samples = mf(rx_samples)

            @savefig sdr_MSK_7.svg
            plt.figure(figsize=(8, 6)); \
            sdr.plot.eye(mf_samples[10*msk.sps : -10*msk.sps], msk.sps); \
            plt.suptitle("Noisy received and matched filtered signal");

        Matched filter and demodulate. Note, the first symbol has $Q = 0$ and the last symbol has $I = 0$.

        .. ipython:: python

            rx_symbols, rx_complex_symbols, _ = msk.demodulate(rx_samples)

            # The symbol decisions are error-free
            np.array_equal(symbols, rx_symbols)

            @savefig sdr_MSK_8.svg
            plt.figure(); \
            sdr.plot.constellation(rx_complex_symbols);

        See the :ref:`psk` example.

    Group:
        modulation-continuous-phase
    """

    def __init__(
        self,
        phase_offset: float = 45,
        symbol_labels: Literal["bin", "gray"] | npt.ArrayLike = "gray",
        sps: int = 8,
    ):
        r"""
        Creates a new MSK object.

        Arguments:
            phase_offset: The absolute phase offset $\phi$ in degrees.
            symbol_labels: The decimal symbol labels of consecutive complex symbols.

                - `"bin"`: The symbols are binary-coded. Adjacent symbols may differ by more than one bit.
                - `"gray":` The symbols are Gray-coded. Adjacent symbols only differ by one bit.
                - `npt.ArrayLike`: An $4$-length array whose indices are the default symbol labels and whose values are
                  the new symbol labels. The default symbol labels are $0$ to $4-1$ for phases starting at $1 + 0j$
                  and going counter-clockwise around the unit circle.

            sps: The number of samples per symbol $f_s / f_{sym}$.

        See Also:
            sdr.half_sine
        """
        verify_scalar(phase_offset, float=True)
        verify_scalar(sps, int=True, positive=True, even=True)

        pulse_shape = half_sine(sps)

        super().__init__(
            phase_offset=phase_offset,
            symbol_labels=symbol_labels,
            sps=sps,
            pulse_shape=pulse_shape,
        )
