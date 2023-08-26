"""
A module containing various discrete-time pulse shapes.
"""
from __future__ import annotations

import numpy as np
from typing_extensions import Literal

from .._helper import export


@export
def rectangular(
    sps: int,
    span: int = 1,
    norm: Literal["power", "energy", "passband"] = "energy",
) -> np.ndarray:
    r"""
    Returns a rectangular pulse shape.

    Arguments:
        sps: The number of samples per symbol.
        span: The length of the filter in symbols. The length of the filter is `span * sps` samples,
            but only the center `sps` samples are non-zero. The only reason for `span` to be larger than 1 is to
            add delay to the filter.
        norm: Indicates how to normalize the pulse shape.

            - `"power"`: The pulse shape is normalized so that the maximum power is 1.
            - `"energy"`: The pulse shape is normalized so that the total energy is 1.
            - `"passband"`: The pulse shape is normalized so that the passband gain is 1.

    Returns:
        The rectangular pulse shape.

    Examples:
        .. ipython:: python

            h_rect = sdr.rectangular(10)

            @savefig sdr_rectangular_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_rect);

            @savefig sdr_rectangular_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_rect);

        See the :ref:`pulse-shapes` example.

    Group:
        modulation-pulse-shaping
    """
    if not isinstance(sps, int):
        raise TypeError(f"Argument 'sps' must be an integer, not {type(sps)}.")
    if not sps >= 1:
        raise ValueError(f"Argument 'sps' must be at least 1, not {sps}.")

    if not isinstance(span, int):
        raise TypeError(f"Argument 'span' must be an integer, not {type(span)}.")
    if not span >= 1:
        raise ValueError(f"Argument 'span' must be at least 1, not {span}.")

    length = span * sps
    h = np.zeros(length, dtype=float)
    idx = (length - sps) // 2
    h[idx : idx + sps] = 1

    h = _normalize(h, norm)

    return h


@export
def half_sine(
    sps: int,
    span: int = 1,
    norm: Literal["power", "energy", "passband"] = "energy",
) -> np.ndarray:
    r"""
    Returns a half-sine pulse shape.

    Arguments:
        sps: The number of samples per symbol.
        span: The length of the filter in symbols. The length of the filter is `span * sps` samples,
            but only the center `sps` samples are non-zero. The only reason for `span` to be larger than 1 is to
            add delay to the filter.
        norm: Indicates how to normalize the pulse shape.

            - `"power"`: The pulse shape is normalized so that the maximum power is 1.
            - `"energy"`: The pulse shape is normalized so that the total energy is 1.
            - `"passband"`: The pulse shape is normalized so that the passband gain is 1.

    Returns:
        The half-sine pulse shape.

    Examples:
        .. ipython:: python

            h_half_sine = sdr.half_sine(10)

            @savefig sdr_half_sine_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_half_sine);

            @savefig sdr_half_sine_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_half_sine);

        See the :ref:`pulse-shapes` example.

    Group:
        modulation-pulse-shaping
    """
    if not isinstance(sps, int):
        raise TypeError(f"Argument 'sps' must be an integer, not {type(sps)}.")
    if not sps >= 1:
        raise ValueError(f"Argument 'sps' must be at least 1, not {sps}.")

    if not isinstance(span, int):
        raise TypeError(f"Argument 'span' must be an integer, not {type(span)}.")
    if not span >= 1:
        raise ValueError(f"Argument 'span' must be at least 1, not {span}.")

    length = span * sps
    h = np.zeros(length, dtype=float)
    idx = (length - sps) // 2
    h[idx : idx + sps] = np.sin(np.pi * np.arange(sps) / sps)

    h = _normalize(h, norm)

    return h


@export
def gaussian(
    time_bandwidth: float,
    span: int,
    sps: int,
    norm: Literal["power", "energy", "passband"] = "passband",
) -> np.ndarray:
    r"""
    Returns a Gaussian pulse shape.

    Arguments:
        time_bandwidth: The time-bandwidth product $B T_{sym}$ of the filter, where $B$ is the one-sided
            3-dB bandwidth in Hz and $T_{sym}$ is the symbol time in seconds. The time-bandwidth product
            can also be thought of as the fractional bandwidth $B / f_{sym}$. Smaller values produce
            wider pulses.
        span: The length of the filter in symbols. The length of the filter is `span * sps + 1` samples.
            The filter order `span * sps` must be even.
        sps: The number of samples per symbol.
        norm: Indicates how to normalize the pulse shape.

            - `"power"`: The pulse shape is normalized so that the maximum power is 1.
            - `"energy"`: The pulse shape is normalized so that the total energy is 1.
            - `"passband"`: The pulse shape is normalized so that the passband gain is 1.

    Returns:
        The Gaussian pulse shape.

    Notes:
        The Gaussian pulse shape has a transfer function of

        $$H(f) = \exp(-\alpha^2 f^2) .$$

        The parameter $\alpha$ is related to the 3-dB bandwidth $B$ by

        $$\alpha = \sqrt{\frac{\ln 2}{2}} \frac{T_{sym}}{B T_{sym}} = \sqrt{\frac{\ln 2}{2}} \frac{1}{B}.$$

        The impulse response is defined as

        $$h(t) = \frac{\sqrt{\pi}}{\alpha} \exp\left[-\left(\frac{\pi}{\alpha} t\right)^2 \right]. $$

    References:
        - https://www.mathworks.com/help/signal/ref/gaussdesign.html
        - https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470041956.app2

    Examples:
        .. ipython:: python

            h_0p1 = sdr.gaussian(0.1, 5, 10); \
            h_0p2 = sdr.gaussian(0.2, 5, 10); \
            h_0p3 = sdr.gaussian(0.3, 5, 10);

            @savefig sdr_gaussian_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_0p1, label=r"$B T_{sym} = 0.1$"); \
            sdr.plot.impulse_response(h_0p2, label=r"$B T_{sym} = 0.2$"); \
            sdr.plot.impulse_response(h_0p3, label=r"$B T_{sym} = 0.3$")

        See the :ref:`pulse-shapes` example.

    Group:
        modulation-pulse-shaping
    """
    if not isinstance(time_bandwidth, (int, float)):
        raise TypeError(f"Argument 'time_bandwidth' must be a number, not {type(time_bandwidth)}.")
    if not time_bandwidth > 0:
        raise ValueError(f"Argument 'time_bandwidth' must be greater than 0, not {time_bandwidth}.")

    if not isinstance(span, int):
        raise TypeError(f"Argument 'span' must be an integer, not {type(span)}.")
    if not span > 1:
        raise ValueError(f"Argument 'span' must be greater than 1, not {span}.")

    if not isinstance(sps, int):
        raise TypeError(f"Argument 'sps' must be an integer, not {type(sps)}.")
    if not sps > 1:
        raise ValueError(f"Argument 'sps' must be greater than 1, not {sps}.")

    if not span * sps % 2 == 0:
        raise ValueError("The order of the filter (span * sps) must be even.")

    t = np.arange(-(span * sps) // 2, (span * sps) // 2 + 1, dtype=float)
    t /= sps

    # Equation B.2
    Ts = 1
    alpha = np.sqrt(np.log(2) / 2) * Ts / time_bandwidth

    # Equation B.3
    h = np.sqrt(np.pi) / alpha * np.exp(-((np.pi * t / alpha) ** 2))

    h = _normalize(h, norm)

    return h


@export
def raised_cosine(
    alpha: float,
    span: int,
    sps: int,
    norm: Literal["power", "energy", "passband"] = "energy",
) -> np.ndarray:
    r"""
    Returns a raised cosine (RC) pulse shape.

    Arguments:
        alpha: The excess bandwidth $0 \le \alpha \le 1$ of the filter.
        span: The length of the filter in symbols. The length of the filter is `span * sps + 1` samples.
            The filter order `span * sps` must be even.
        sps: The number of samples per symbol.
        norm: Indicates how to normalize the pulse shape.

            - `"power"`: The pulse shape is normalized so that the maximum power is 1.
            - `"energy"`: The pulse shape is normalized so that the total energy is 1.
            - `"passband"`: The pulse shape is normalized so that the passband gain is 1.

    Returns:
        The raised cosine pulse shape.

    References:
        - Michael Rice, *Digital Communications: A Discrete Time Approach*, Appendix A.

    Examples:
        The excess bandwidth $\alpha$ controls bandwidth of the filter. A smaller $\alpha$ results in a
        narrower bandwidth at the expense of higher sidelobes.

        .. ipython:: python

            h_0p1 = sdr.raised_cosine(0.1, 8, 10); \
            h_0p5 = sdr.raised_cosine(0.5, 8, 10); \
            h_0p9 = sdr.raised_cosine(0.9, 8, 10)

            @savefig sdr_raised_cosine_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_0p1, label=r"$\alpha = 0.1$"); \
            sdr.plot.impulse_response(h_0p5, label=r"$\alpha = 0.5$"); \
            sdr.plot.impulse_response(h_0p9, label=r"$\alpha = 0.9$")

            @savefig sdr_raised_cosine_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_0p1, label=r"$\alpha = 0.1$"); \
            sdr.plot.magnitude_response(h_0p5, label=r"$\alpha = 0.5$"); \
            sdr.plot.magnitude_response(h_0p9, label=r"$\alpha = 0.9$")

        The span of the filter affects the stopband attenuation. A longer span results in greater stopband
        attenuation and lower sidelobes.

        .. ipython:: python

            h_4 = sdr.raised_cosine(0.1, 4, 10); \
            h_8 = sdr.raised_cosine(0.1, 8, 10); \
            h_16 = sdr.raised_cosine(0.1, 16, 10)

            @savefig sdr_raised_cosine_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_4, label="span = 4"); \
            sdr.plot.impulse_response(h_8, label="span = 8"); \
            sdr.plot.impulse_response(h_16, label="span = 16")

            @savefig sdr_raised_cosine_4.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_4, label="span = 4"); \
            sdr.plot.magnitude_response(h_8, label="span = 8"); \
            sdr.plot.magnitude_response(h_16, label="span = 16")

        See the :ref:`pulse-shapes` example.

    Group:
        modulation-pulse-shaping
    """
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"Argument 'alpha' must be a number, not {type(alpha)}.")
    if not 0 <= alpha <= 1:
        raise ValueError(f"Argument 'alpha' must be between 0 and 1, not {alpha}.")

    if not isinstance(span, int):
        raise TypeError(f"Argument 'span' must be an integer, not {type(span)}.")
    if not span > 1:
        raise ValueError(f"Argument 'span' must be greater than 1, not {span}.")

    if not isinstance(sps, int):
        raise TypeError(f"Argument 'sps' must be an integer, not {type(sps)}.")
    if not sps > 1:
        raise ValueError(f"Argument 'sps' must be greater than 1, not {sps}.")

    if not span * sps % 2 == 0:
        raise ValueError("The order of the filter (span * sps) must be even.")

    t = np.arange(-(span * sps) // 2, (span * sps) // 2 + 1, dtype=float)
    Ts = sps

    # Handle special cases where the denominator is zero
    t[t == 0] += 1e-16
    if alpha > 0:
        t1 = Ts / (2 * alpha)
        t[t == -t1] -= 1e-8
        t[t == t1] += 1e-8

    # Equation A-27
    A = np.sin(np.pi * t / Ts) / (np.pi * t / Ts)
    B = np.cos(np.pi * alpha * t / Ts) / (1 - (2 * alpha * t / Ts) ** 2)
    h = A * B

    h = _normalize(h, norm)

    return h


@export
def root_raised_cosine(
    alpha: float,
    span: int,
    sps: int,
    norm: Literal["power", "energy", "passband"] = "energy",
) -> np.ndarray:
    r"""
    Returns a square root raised cosine (SRRC) pulse shape.

    Arguments:
        alpha: The excess bandwidth $0 \le \alpha \le 1$ of the filter.
        span: The length of the filter in symbols. The length of the filter is `span * sps + 1` samples.
            The filter order `span * sps` must be even.
        sps: The number of samples per symbol.
        norm: Indicates how to normalize the pulse shape.

            - `"power"`: The pulse shape is normalized so that the maximum power is 1.
            - `"energy"`: The pulse shape is normalized so that the total energy is 1.
            - `"passband"`: The pulse shape is normalized so that the passband gain is 1.

    Returns:
        The square-root raised cosine pulse shape.

    References:
        - Michael Rice, *Digital Communications: A Discrete Time Approach*, Appendix A.

    Examples:
        The excess bandwidth $\alpha$ controls bandwidth of the filter. A smaller $\alpha$ results in a
        narrower bandwidth at the expense of higher sidelobes.

        .. ipython:: python

            h_0p1 = sdr.root_raised_cosine(0.1, 8, 10); \
            h_0p5 = sdr.root_raised_cosine(0.5, 8, 10); \
            h_0p9 = sdr.root_raised_cosine(0.9, 8, 10)

            @savefig sdr_root_raised_cosine_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_0p1, label=r"$\alpha = 0.1$"); \
            sdr.plot.impulse_response(h_0p5, label=r"$\alpha = 0.5$"); \
            sdr.plot.impulse_response(h_0p9, label=r"$\alpha = 0.9$")

            @savefig sdr_root_raised_cosine_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_0p1, label=r"$\alpha = 0.1$"); \
            sdr.plot.magnitude_response(h_0p5, label=r"$\alpha = 0.5$"); \
            sdr.plot.magnitude_response(h_0p9, label=r"$\alpha = 0.9$")

        The span of the filter affects the stopband attenuation. A longer span results in greater stopband
        attenuation and lower sidelobes.

        .. ipython:: python

            h_4 = sdr.root_raised_cosine(0.1, 4, 10); \
            h_8 = sdr.root_raised_cosine(0.1, 8, 10); \
            h_16 = sdr.root_raised_cosine(0.1, 16, 10)

            @savefig sdr_root_raised_cosine_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h_4, label="span = 4"); \
            sdr.plot.impulse_response(h_8, label="span = 8"); \
            sdr.plot.impulse_response(h_16, label="span = 16")

            @savefig sdr_root_raised_cosine_4.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h_4, label="span = 4"); \
            sdr.plot.magnitude_response(h_8, label="span = 8"); \
            sdr.plot.magnitude_response(h_16, label="span = 16")

        See the :ref:`pulse-shapes` example.

    Group:
        modulation-pulse-shaping
    """
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"Argument 'alpha' must be a number, not {type(alpha)}.")
    if not 0 <= alpha <= 1:
        raise ValueError(f"Argument 'alpha' must be between 0 and 1, not {alpha}.")

    if not isinstance(span, int):
        raise TypeError(f"Argument 'span' must be an integer, not {type(span)}.")
    if not span > 1:
        raise ValueError(f"Argument 'span' must be greater than 1, not {span}.")

    if not isinstance(sps, int):
        raise TypeError(f"Argument 'sps' must be an integer, not {type(sps)}.")
    if not sps > 1:
        raise ValueError(f"Argument 'sps' must be greater than 1, not {sps}.")

    if not span * sps % 2 == 0:
        raise ValueError("The order of the filter (span * sps) must be even.")

    t = np.arange(-(span * sps) // 2, (span * sps) // 2 + 1, dtype=float)
    Ts = sps  # Symbol duration (in samples)

    # Handle special cases where the denominator is zero
    t[t == 0] += 1e-16
    if alpha > 0:
        t1 = Ts / (4 * alpha)
        t[t == -t1] -= 1e-8
        t[t == t1] += 1e-8

    # Equation A-30
    A = np.sin(np.pi * (1 - alpha) * t / Ts)
    B = 4 * alpha * t / Ts * np.cos(np.pi * (1 + alpha) * t / Ts)
    C = np.pi * t / Ts * (1 - (4 * alpha * t / Ts) ** 2)
    h = (A + B) / C
    h /= 1 / np.sqrt(Ts)

    h = _normalize(h, norm)

    return h


def _normalize(h: np.ndarray, norm: Literal["power", "energy", "passband"]) -> np.ndarray:
    if norm == "power":
        h /= np.sqrt(np.max(np.abs(h) ** 2))
    elif norm == "energy":
        h /= np.sqrt(np.sum(np.abs(h) ** 2))
    elif norm == "passband":
        h /= np.sum(h)
    else:
        raise ValueError(f"Argument 'norm' must be 'power', 'energy', or 'passband', not {norm}.")

    return h
