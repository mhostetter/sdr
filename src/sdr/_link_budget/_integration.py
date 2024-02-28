"""
A module containing functions related to coherent gain loss.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._conversion import db
from .._helper import export


@export
def coherent_gain_loss(
    integration_time: npt.ArrayLike,
    freq_offset: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    r"""
    Computes the coherent gain loss (CGL) as a function of the given integration time and frequency offset.

    Arguments:
        integration_time: The coherent integration time $T_c$ in seconds.
        freq_offset: The frequency offset $\Delta f$ in Hz.

    Returns:
        The coherent gain loss (CGL) in dB.

    Notes:
        $$\text{CGL} = 10 \log_{10} \left( \text{sinc}^2 \left( T_c \Delta f \right) \right)$$
        $$\text{sinc}(x) = \frac{\sin(\pi x)}{\pi x}$$

    Examples:
        Compute the coherent gain loss for an integration time of 1 ms and a frequency offset of 235 Hz.

        .. ipython:: python

            sdr.coherent_gain_loss(1e-3, 235)

        Compute the coherent gain loss for an integration time of 1 ms and an array of frequency offsets.

        .. ipython:: python

            sdr.coherent_gain_loss(1e-3, [0, 100, 200, 300, 400, 500])

        Plot coherent gain loss as a function of frequency offset for an integration time of 1 ms.

        .. ipython:: python

            f = np.linspace(0, 5e3, 1001)
            cgl = sdr.coherent_gain_loss(1e-3, f)

            @savefig sdr_coherent_gain_loss_1.png
            plt.figure(); \
            plt.plot(f, cgl); \
            plt.ylim(-50, 10); \
            plt.xlabel("Frequency offset (Hz)"); \
            plt.ylabel("Coherent gain loss (dB)"); \
            plt.title("Coherent gain loss for a 1-ms integration");

        Plot coherent gain loss as a function of integration time for a frequency offset of 235 Hz.

        .. ipython:: python

            t = np.linspace(0, 2e-2, 1001)
            cgl = sdr.coherent_gain_loss(t, 235)

            @savefig sdr_coherent_gain_loss_2.png
            plt.figure(); \
            plt.plot(t * 1e3, cgl); \
            plt.ylim(-50, 10); \
            plt.xlabel("Integration time (ms)"); \
            plt.ylabel("Coherent gain loss (dB)"); \
            plt.title("Coherent gain loss for a 235-Hz frequency offset");

    Group:
        link-budget-coherent-integration
    """
    integration_time = np.asarray(integration_time)
    freq_offset = np.asarray(freq_offset)

    cgl = np.sinc(integration_time * freq_offset) ** 2
    cgl_db = db(cgl)

    return cgl_db
