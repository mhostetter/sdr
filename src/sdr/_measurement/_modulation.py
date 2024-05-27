"""
A module containing measurement functions related to modulation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.integrate
from typing_extensions import Literal

from .._helper import export
from ._power import average_power


@export
def evm(
    x_hat: npt.ArrayLike,
    ref: npt.ArrayLike,
    norm: Literal["average-power-ref", "average-power", "peak-power"] = "average-power-ref",
    output: Literal["rms", "all"] | float = "rms",
) -> float:
    r"""
    Measures the error-vector magnitude (EVM) of the complex symbols $\hat{x}[k]$.

    Arguments:
        x_hat: The complex symbols $\hat{x}[k]$ to be measured.
        ref: The complex reference symbols $x[k]$. This can be the noiseless transmitted symbols or the
            modulation's symbol map.
        norm: The normalization source used in the EVM calculation.

            - `"average-power-ref"`: The average power of the reference symbols $x[k]$.

            $$P_{\text{ref}} = \frac{1}{N} \sum_{k=0}^{N-1} \left| x[k] \right|^2$$

            - `"average-power"`: The average power of the received symbols $\hat{x}[k]$.

            $$P_{\text{ref}} = \frac{1}{N} \sum_{k=0}^{N-1} \left| \hat{x}[k] \right|^2$$

            - `"peak-power"`: The peak power of the received symbols $\hat{x}[k]$.

            $$P_{\text{ref}} = \text{max} \left| \hat{x}[k] \right|^2$$

        output: The output type of the EVM calculation.

            - `"rms"`: The root-mean-square (RMS) EVM.

            $$
            \text{EVM}_{\text{RMS}} =
            100 \sqrt{\frac{\frac{1}{N} \sum_{k=0}^{N-1} \left| \hat{x}[k] - x[k] \right|^2}{P_{\text{ref}}}}
            $$

            - `"all"`: The instantaneous EVM for each symbol.

            $$
            \text{EVM}_{k} =
            100 \sqrt{\frac{\left| \hat{x}[k] - x[k] \right|^2}{P_{\text{ref}}}}
            $$

            - `float`: The RMS EVM for the given percentile (0 - 100).

    Examples:
        Create QPSK symbols with $E_s/N_0$ of 20 dB.

        .. ipython:: python

            psk = sdr.PSK(4, phase_offset=45); \
            s = np.random.randint(0, psk.order, 1000); \
            x = psk.map_symbols(s); \
            x_hat = sdr.awgn(x, 20)

            @savefig sdr_evm_1.png
            plt.figure(); \
            sdr.plot.constellation(x_hat, label=r"$\hat{x}[k]$"); \
            sdr.plot.symbol_map(psk.symbol_map, label=r"Reference"); \
            plt.title("QPSK Constellation at 20 dB $E_s/N_0$");

        Measure the RMS EVM, normalizing with the average power of the reference symbols.
        Either the symbol map or noiseless transmitted symbols may be passed.

        .. ipython:: python

            sdr.evm(x_hat, psk.symbol_map)
            sdr.evm(x_hat, x)

        Measure the RMS EVM, normalizing with the average power of the received symbols.

        .. ipython:: python

            sdr.evm(x_hat, psk.symbol_map, norm="average-power")

        Measure the RMS EVM, normalizing with the peak power of the received symbols.

        .. ipython:: python

            sdr.evm(x_hat, psk.symbol_map, norm="peak-power")

        Measure the 95th percentile EVM.

        .. ipython:: python

            sdr.evm(x_hat, psk.symbol_map, output=95)

        Measure the instantaneous EVM for each symbol.

        .. ipython:: python

            inst_evm = sdr.evm(x_hat, psk.symbol_map, output="all")

            @savefig sdr_evm_2.png
            plt.figure(); \
            plt.hist(inst_evm, bins=20); \
            plt.xlabel("RMS EVM (%)"); \
            plt.ylabel("Count"); \
            plt.title("EVM Histogram");

    Group:
        measurement-modulation
    """
    x_hat = np.asarray(x_hat)
    ref = np.asarray(ref)

    if norm == "average-power-ref":
        ref_power = average_power(ref)
    elif norm == "average-power":
        ref_power = average_power(x_hat)
    elif norm == "peak-power":
        ref_power = np.max(np.abs(x_hat) ** 2)
    else:
        raise ValueError(
            f"Argument 'norm' must be one of 'average-power-ref', 'average-power', or 'peak-power', not {ref}."
        )

    if ref.shape == x_hat.shape:
        # Compute the error vectors to each reference symbol
        error_vectors = x_hat - ref
    else:
        # The reference symbols are the symbol map. We must first determine the most likely reference symbol for each
        # received symbol. Then we must compute the error vectors to those symbols.
        symbol_map = ref
        all_error_vectors = np.subtract.outer(x_hat, symbol_map)
        s_hat = np.argmin(np.abs(all_error_vectors), axis=-1)
        error_vectors = x_hat - symbol_map[s_hat]

    if output == "rms":
        rms_evm = 100 * np.sqrt(average_power(error_vectors) / ref_power)
        return rms_evm

    inst_evm = 100 * np.sqrt(np.abs(error_vectors) ** 2 / ref_power)
    if output == "all":
        return inst_evm
    if isinstance(output, (int, float)) and 0 <= output <= 100:
        perc_evm = np.percentile(inst_evm, output)
        return perc_evm

    raise ValueError(f"Argument 'output' must be 'rms' or a float between 0 and 100, not {output}.")


@export
def rms_bandwidth(x: npt.ArrayLike, sample_rate: float = 1.0) -> float:
    r"""
    Measures the RMS bandwidth $B_{\text{rms}}$ of the signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$ to measure.

            .. note::
                For best measurement performance, the time-domain signal should be arbitrarily long. This allows for
                more averaging of its power spectra.

        sample_rate: The sample rate $f_s$ in samples/s.

    Returns:
        The RMS signal bandwidth $B_{\text{rms}}$ in Hz.

    Notes:
        The root-mean-square (RMS) bandwidth $B_{\text{rms}}$ is calculated by

        $$B_{\text{rms}} = \sqrt{\frac{\int_{-\infty}^{\infty} (f - \mu_f)^2 S(f - \mu_f) \, df}{\int_{-\infty}^{\infty} S(f - \mu_f) \, df}}$$

        where $S(f)$ is the power spectral density (PSD) of the signal $x[n]$. The RMS bandwidth is measured about the
        centroid of the spectrum

        $$\mu_f = \frac{\int_{-\infty}^{\infty} f S(f) \, df}{\int_{-\infty}^{\infty} S(f) \, df} .$$

        The RMS bandwidth is a measure of the spread of the spectrum about the centroid. For a rectangular spectrum,
        the RMS bandwidth is $B_{\text{rms}} = B_s / \sqrt{12}$.

    Examples:
        Calculate the RMS bandwidth of a signal with a rectangular spectrum with normalized bandwidth of 1.

        .. ipython:: python

            symbol_rate = 1  # symbols/s
            symbol_rate / np.sqrt(12)

        Create a BPSK signal with a rectangular pulse shape. Note, the time-domain pulse shape is rectangular, but the
        spectrum is sinc-shaped. Measure the RMS bandwidth of the signal and compare it to the ideal rectangular
        spectrum.

        .. ipython:: python

            psk = sdr.PSK(2, pulse_shape="rect")
            symbols = np.random.randint(0, psk.order, 10_000)
            x_rect = psk.modulate(symbols)
            sdr.rms_bandwidth(x_rect, sample_rate=symbol_rate * psk.sps)

        Make the same measurements with square-root raised cosine (SRRC) pulse shaping. The SRRC spectrum is narrower
        and, therefore, closer to the rectangular spectrum.

        .. ipython:: python

            psk = sdr.PSK(2, pulse_shape="srrc")
            symbols = np.random.randint(0, psk.order, 10_000)
            x_srrc = psk.modulate(symbols)
            sdr.rms_bandwidth(x_srrc, sample_rate=symbol_rate * psk.sps)

        Plot the power spectral density (PSD) of the rectangular and SRRC pulse-shaped signals.

        .. ipython:: python

            @savefig sdr_rms_bandwidth_1.png
            plt.figure(); \
            sdr.plot.periodogram(x_rect, sample_rate=symbol_rate * psk.sps, label="Rectangular"); \
            sdr.plot.periodogram(x_srrc, sample_rate=symbol_rate * psk.sps, label="SRRC");

    Group:
        measurement-modulation
    """
    x = np.asarray(x)

    f, psd = scipy.signal.welch(
        x, fs=sample_rate, detrend=False, return_onesided=False, scaling="density", average="mean"
    )

    # Shift so that the frequencies increase monotonically
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)

    # Calculate the centroid of the PSD
    f_mean = scipy.integrate.simps(f * psd, f) / scipy.integrate.simps(psd, f)
    f -= f_mean

    # Calculate the RMS bandwidth
    ms_bandwidth = scipy.integrate.simps(f**2 * psd, f) / scipy.integrate.simps(psd, f)
    rms_bandwidth = np.sqrt(float(ms_bandwidth))

    return rms_bandwidth
