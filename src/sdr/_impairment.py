"""
A module containing functions for simulating various signal impairments.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._helper import export
from ._measurement import average_power


@export
def awgn(
    x: npt.ArrayLike,
    snr: float | None = None,
    noise: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    r"""
    Adds additive white Gaussian noise (AWGN) to the time-domain signal $x[n]$.

    Arguments:
        x: The time-domain signal $x[n]$ to which AWGN is added.
        snr: The desired signal-to-noise ratio (SNR) in dB. If specified, the average signal power is measured
            explicitly. It is assumed that $x[n]$ contains signal only. If the signal power is known, the
            desired noise variance can be computed and passed in `noise`. If `snr` is `None`,
            `noise` must be specified.
        noise: The noise power (variance) in linear units. If `noise` is `None`, `snr` must be specified.
        seed: The seed for the random number generator. This is passed to :func:`numpy.random.default_rng()`.

    Returns:
        The noisy signal $x[n] + w[n]$.

    Notes:
        The signal-to-noise ratio (SNR) is defined as

        $$
        \text{SNR} = \frac{P_{\text{signal,avg}}}{P_{\text{noise}}}
        = \frac{\frac{1}{N} \sum_{n=0}^{N-1} \left| x[n] \right|^2}{\sigma^2} ,
        $$

        where $\sigma^2$ is the noise variance. The output signal, with the specified SNR, is $y[n] = x[n] + w[n]$.

        For real signals:
        $$w \sim \mathcal{N}(0, \sigma^2)$$

        For complex signals:
        $$w \sim \mathcal{CN}(0, \sigma^2) = \mathcal{N}(0, \sigma^2 / 2) + j\mathcal{N}(0, \sigma^2 / 2)$$

    Examples:
        .. ipython:: python

            x = np.sin(2 * np.pi * 5 * np.arange(100) / 100); \
            y = sdr.awgn(x, snr=10)

            @savefig sdr_awgn_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, label="$x[n]$"); \
            sdr.plot.time_domain(y, label="$y[n]$"); \
            plt.title("Input signal $x[n]$ and noisy output signal $y[n]$ with 10 dB SNR"); \
            plt.tight_layout(); \
            plt.show()

    Group:
        impairments
    """
    x = np.asarray(x)
    if snr:
        snr_linear = 10 ** (snr / 10)
        signal_power = average_power(x)
        noise_power = signal_power / snr_linear
    elif noise:
        noise_power = noise
    else:
        raise ValueError("Either 'snr' or 'noise' must be specified.")

    rng = np.random.default_rng(seed)
    if np.iscomplexobj(x):
        w = rng.normal(0, np.sqrt(noise_power / 2), x.shape) + 1j * rng.normal(0, np.sqrt(noise_power / 2), x.shape)
    else:
        w = rng.normal(0, np.sqrt(noise_power), x.shape)

    return x + w
