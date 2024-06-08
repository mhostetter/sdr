"""
A module containing functions related to non-coherent integration.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.optimize
from typing_extensions import Literal

from .._conversion import db
from .._helper import export
from ._theory import p_d


@export
def non_coherent_gain(
    n_nc: npt.ArrayLike,
    snr: npt.ArrayLike,
    p_fa: npt.ArrayLike = 1e-6,
    detector: Literal["linear", "square-law"] = "square-law",
    complex: bool = True,
    snr_ref: Literal["input", "output"] = "input",
    extrapolate: bool = True,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the SNR improvement by non-coherently integrating $N_{nc}$ samples.

    Arguments:
        n_nc: The number of samples $N_{nc}$ to non-coherently integrate.
        snr: The reference SNR in dB.
        p_fa: The desired probability of false alarm $P_{fa}$. This is used to compute the necessary thresholds before
            and after integration. The non-coherent gain is slightly affected by the $P_{fa}$.
        detector: The detector type.

            - `"coherent"`: A coherent detector, $$T(x) = \mathrm{Re}\left\{\sum_{i=0}^{N_c-1} x[n-i]\right\} .$$
            - `"linear"`: A linear detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right| .$$
            - `"square-law"`: A square-law detector, $$T(x) = \sum_{j=0}^{N_{nc}-1}\left|\sum_{i=0}^{N_c-1} x[n-i-jN_c]\right|^2 .$$

        complex: Indicates whether the input signal is real or complex. This affects how the SNR is converted
            to noise variance.
        snr_ref: The SNR reference.

            - `"input"`: The SNR is referenced at the input of the non-coherent integrator.
            - `"output"`: The SNR is referenced at the output of the non-coherent integrator.

        extrapolate: Indicates whether to extrapolate $G_{nc}$ using smaller values of $N_{nc}$. This is only done when
            the non-coherent gain cannot be explicitly solved for due to lack of floating-point precision.
            If `False`, the function will return `np.nan` for any $N_{nc}$ that cannot be solved for.

    Returns:
        The non-coherent gain $G_{nc}$ in dB.

    See Also:
        sdr.peebles

    Notes:
        $$y[m] = \sum_{n=0}^{N_{nc}-1} \left| x[m-n] \right|^2$$
        $$\text{SNR}_{y,\text{dB}} = \text{SNR}_{x,\text{dB}} + G_{nc}$$

    Examples:
        See the :ref:`non-coherent-integration` example.

        Compute the non-coherent gain for various integration lengths at 10-dB SNR.

        .. ipython:: python

            sdr.non_coherent_gain(1, 10)
            sdr.non_coherent_gain(2, 10)
            sdr.non_coherent_gain(10, 10)
            sdr.non_coherent_gain(20, 10)

        Plot the non-coherent gain parameterized by input SNR.

        .. ipython:: python

            plt.figure(); \
            n = np.logspace(0, 3, 51).astype(int); \
            plt.semilogx(n, sdr.coherent_gain(n), color="k");
            for snr in np.arange(-30, 40, 10):
                plt.semilogx(n, sdr.non_coherent_gain(n, snr, snr_ref="input"), label=f"{snr} dB")
            @savefig sdr_non_coherent_gain_1.png
            plt.legend(title="Input SNR", loc="upper left"); \
            plt.xlabel("Number of samples, $N_{nc}$"); \
            plt.ylabel("Non-coherent gain, $G_{nc}$"); \
            plt.title("Non-coherent gain for various input SNRs");

        Compare the non-coherent gain for linear and square-law detectors for various input SNRs. Notice that the
        square-law detector performs better at low SNRs and the linear detector performs better at high SNRs.

        .. ipython:: python

            @savefig sdr_non_coherent_gain_2.png
            plt.figure(); \
            snr = np.linspace(-20, 10, 21); \
            plt.plot(snr, sdr.non_coherent_gain(10, snr, p_fa=1e-6, detector="linear"), label="Linear"); \
            plt.plot(snr, sdr.non_coherent_gain(10, snr, p_fa=1e-6, detector="square-law"), label="Square-law"); \
            plt.legend(title="Detector", loc="upper left"); \
            plt.xlabel("Input signal-to-noise ratio (dB)"); \
            plt.ylabel("Non-coherent gain, $G_{nc}$"); \
            plt.title("Non-coherent gain for $N_{nc} = 10$");

        Examine the non-coherent gain across input SNR and false alarm rate for non-coherently integrating 10 samples.
        Notice that the non-coherent gain is affected by both. The coherent integration gain, however, is
        a constant 10 dB across both.

        .. ipython:: python

            plt.figure(); \
            snr = np.linspace(-40, 12, 101); \
            n_nc = 10;
            for p_fa in [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
                g_nc = sdr.non_coherent_gain(n_nc, snr, p_fa)
                plt.plot(snr, g_nc, label=f"{p_fa:1.0e}")
            @savefig sdr_non_coherent_gain_3.png
            plt.legend(title="$P_{fa}$"); \
            plt.ylim(0, 10); \
            plt.xlabel("Input signal-to-noise ratio (dB)"); \
            plt.ylabel("Non-coherent gain (dB)"); \
            plt.title("Non-coherent gain for $N_{nc} = 10$");

    Group:
        detection-non-coherent-integration
    """
    n_nc = np.asarray(n_nc, dtype=object)  # Downstream functions want a Python int
    snr = np.asarray(snr)
    p_fa = np.asarray(p_fa)

    if np.any(n_nc < 1):
        raise ValueError(f"Argument 'n_nc' must be at least 1, not {n_nc}.")
    if np.any(p_fa < 0) or np.any(p_fa > 1):
        raise ValueError(f"Argument 'p_fa' must be between 0 and 1, not {p_fa}.")
    if snr_ref not in ["input", "output"]:
        raise ValueError(f"Argument 'snr_ref' must be either 'input' or 'output', not {snr_ref}.")

    if snr_ref == "input":
        g_nc = _non_coherent_gain_in(n_nc, snr, p_fa, detector, complex)
    else:
        g_nc = _non_coherent_gain_out(n_nc, snr, p_fa, detector, complex)

    if extrapolate:
        g_nc = _extrapolate_non_coherent_gain(n_nc, snr, p_fa, g_nc, snr_ref, detector, complex)

    if g_nc.ndim == 0:
        g_nc = float(g_nc)

    return g_nc


@np.vectorize
def _non_coherent_gain_in(
    n_nc: int, snr: float, p_fa: float, detector: Literal["linear", "square-law"], complex: bool
) -> float:
    """
    Solves for the non-coherent gain when the SNR is referenced at the input of the non-coherent integrator.
    """
    if n_nc == 1:
        return 0.0

    # Compute the theoretical p_d with the following input SNR and n_nc non-coherent combinations
    p_d_in = p_d(snr, p_fa, detector=detector, complex=complex, n_c=1, n_nc=n_nc)
    if p_d_in == 1:
        return np.nan

    def root_eq(snr_out: float):
        p_d_out = p_d(snr_out, p_fa, detector=detector, complex=complex, n_c=1, n_nc=1)
        return db(p_d_out) - db(p_d_in)  # Use logarithms for numerical stability

    # Determine the output SNR that, without non-coherent integration, yields the same probability of detection.
    snr_out = scipy.optimize.brentq(root_eq, snr, snr + db(n_nc))  # Bounds are 0 to coherent gain
    g_nc = snr_out - snr

    return g_nc


@np.vectorize
def _non_coherent_gain_out(
    n_nc: int, snr: float, p_fa: float, detector: Literal["linear", "square-law"], complex: bool
) -> float:
    """
    Solves for the non-coherent gain when the SNR is referenced at the output of the non-coherent integrator.
    """
    if n_nc == 1:
        return 0.0

    # Compute the theoretical p_d with the following output SNR and no non-coherent combinations
    p_d_out = p_d(snr, p_fa, detector=detector, complex=complex, n_c=1, n_nc=1)
    if p_d_out == 1:
        return np.nan

    def root_eq(snr_in: float):
        p_d_in = p_d(snr_in, p_fa, detector=detector, complex=complex, n_c=1, n_nc=n_nc)
        return db(p_d_out) - db(p_d_in)  # Use logarithms for numerical stability

    # Determine the input SNR that, after non-coherent integration, yields the same probability of detection.
    snr_in = scipy.optimize.brentq(root_eq, snr - db(n_nc), snr)  # Bounds are 0 to coherent gain
    g_nc = snr - snr_in

    return g_nc


def _extrapolate_non_coherent_gain(
    n_nc: npt.NDArray[np.float64],
    snr: npt.NDArray[np.float64],
    p_fa: npt.NDArray[np.float64],
    g_nc: npt.NDArray[np.float64],
    snr_ref: Literal["input", "output"],
    detector: Literal["linear", "square-law"],
    complex: bool,
) -> npt.NDArray[np.float64]:
    """
    Computes the non-coherent gain for smaller n_nc values and extrapolates.
    """
    # Broadcast arrays to the same shape and then flatten
    n_nc, snr, p_fa, g_nc = np.broadcast_arrays(n_nc, snr, p_fa, g_nc)
    shape = g_nc.shape
    n_nc = n_nc.ravel()
    snr = snr.ravel()
    p_fa = p_fa.ravel()
    g_nc = g_nc.ravel()

    # Loop through the NaNs and interpolate
    interpolators = {}
    for i in range(g_nc.size):
        if np.isnan(g_nc[i]):
            key = (snr[i], p_fa[i])
            if key not in interpolators:
                n_nc_array = np.logspace(np.log10(max(1, 0.5 * n_nc[i])), np.log10(n_nc[i]), 7).astype(int)
                g_nc_array = non_coherent_gain(n_nc_array, snr[i], p_fa[i], detector, complex, snr_ref, False)

                idxs = ~np.isnan(g_nc_array)
                n_nc_array = n_nc_array[idxs]
                g_nc_array = g_nc_array[idxs]

                if n_nc_array.size == 0:
                    # The gain approaches a limit of the coherent gain of N_nc
                    n_nc_array = np.logspace(0, 2, 5)
                    g_nc_array = db(n_nc_array)

                interpolators[key] = scipy.interpolate.interp1d(
                    np.log10(n_nc_array), g_nc_array, kind="linear", fill_value="extrapolate"
                )

            g_nc[i] = interpolators[key](np.log10(n_nc[i]))

    g_nc = g_nc.reshape(shape)

    return g_nc
