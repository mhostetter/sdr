"""
A module containing various discrete-time pulse shapes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._helper import export, verify_literal, verify_scalar


@export
def rectangular(
    sps: int,
    span: int = 1,
    norm: Literal["power", "energy", "passband"] = "energy",
) -> npt.NDArray[np.float64]:
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

            @savefig sdr_rectangular_1.svg
            plt.figure(); \
            sdr.plot.impulse_response(h_rect);

            @savefig sdr_rectangular_2.svg
            plt.figure(); \
            sdr.plot.magnitude_response(h_rect);

        See the :ref:`pulse-shapes` example.

    Group:
        modulation-pulse-shaping
    """
    verify_scalar(sps, int=True, positive=True)
    verify_scalar(span, int=True, positive=True)
    verify_literal(norm, ["power", "energy", "passband"])

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
) -> npt.NDArray[np.float64]:
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

            @savefig sdr_half_sine_1.svg
            plt.figure(); \
            sdr.plot.impulse_response(h_half_sine);

            @savefig sdr_half_sine_2.svg
            plt.figure(); \
            sdr.plot.magnitude_response(h_half_sine);

        See the :ref:`pulse-shapes` example.

    Group:
        modulation-pulse-shaping
    """
    verify_scalar(sps, int=True, positive=True)
    verify_scalar(span, int=True, positive=True)
    verify_literal(norm, ["power", "energy", "passband"])

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
) -> npt.NDArray[np.float64]:
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
        - `Appendix B: Gaussian Pulse-Shaping Filter.
          <https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470041956.app2>`_

    Examples:
        .. ipython:: python

            h_0p1 = sdr.gaussian(0.1, 5, 10); \
            h_0p2 = sdr.gaussian(0.2, 5, 10); \
            h_0p3 = sdr.gaussian(0.3, 5, 10);

            @savefig sdr_gaussian_1.svg
            plt.figure(); \
            sdr.plot.impulse_response(h_0p1, label=r"$B T_{sym} = 0.1$"); \
            sdr.plot.impulse_response(h_0p2, label=r"$B T_{sym} = 0.2$"); \
            sdr.plot.impulse_response(h_0p3, label=r"$B T_{sym} = 0.3$")

            @savefig sdr_gaussian_2.svg
            plt.figure(); \
            sdr.plot.magnitude_response(h_0p1, label=r"$B T_{sym} = 0.1$"); \
            sdr.plot.magnitude_response(h_0p2, label=r"$B T_{sym} = 0.2$"); \
            sdr.plot.magnitude_response(h_0p3, label=r"$B T_{sym} = 0.3$")

        See the :ref:`pulse-shapes` example.

    Group:
        modulation-pulse-shaping
    """
    verify_scalar(time_bandwidth, float=True, positive=True)
    verify_scalar(span, int=True, positive=True)
    verify_scalar(sps, int=True, inclusive_min=2)
    verify_scalar(span * sps, even=True)
    verify_literal(norm, ["power", "energy", "passband"])

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
) -> npt.NDArray[np.float64]:
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

    Notes:
        The raised cosine pulse shape has a transfer function of

        $$
        H(f) =
        \begin{cases}
        T_{sym},
        & \displaystyle 0 \le \left| f \right| \le \frac{1 - \alpha}{2 T_{sym}} \\
        \displaystyle \frac{T_{sym}}{2} \left[1 + \cos\left(\frac{\pi T_{sym}}{\alpha}\left(\left| f \right| - \frac{1 - \alpha}{2 T_{sym}}\right)\right)\right],
        & \displaystyle \frac{1 - \alpha}{2 T_{sym}} \le \left| f \right| \le \frac{1 + \alpha}{2 T_{sym}} \\
        0,
        & \displaystyle \left| f \right| \ge \frac{1 + \alpha}{2 T_{sym}}
        \end{cases}
        $$

        The impulse response is defined as

        $$
        h(t) =
        \frac
        {\displaystyle \sin\left(\frac{\pi t}{T_{sym}}\right) \cos\left(\frac{\pi \alpha t}{T_{sym}}\right)}
        {\displaystyle \frac{\pi t}{T_{sym}} \left[1 - \left(\frac{2 \alpha t}{T_{sym}} \right)^2 \right]}
        $$

    References:
        - Michael Rice, *Digital Communications: A Discrete Time Approach*, Appendix A.

    Examples:
        The excess bandwidth $\alpha$ controls bandwidth of the filter. A smaller $\alpha$ results in a
        narrower bandwidth at the expense of higher sidelobes.

        .. ipython:: python

            h_0p1 = sdr.raised_cosine(0.1, 8, 10); \
            h_0p5 = sdr.raised_cosine(0.5, 8, 10); \
            h_0p9 = sdr.raised_cosine(0.9, 8, 10)

            @savefig sdr_raised_cosine_1.svg
            plt.figure(); \
            sdr.plot.impulse_response(h_0p1, label=r"$\alpha = 0.1$"); \
            sdr.plot.impulse_response(h_0p5, label=r"$\alpha = 0.5$"); \
            sdr.plot.impulse_response(h_0p9, label=r"$\alpha = 0.9$")

            @savefig sdr_raised_cosine_2.svg
            plt.figure(); \
            sdr.plot.magnitude_response(h_0p1, label=r"$\alpha = 0.1$"); \
            sdr.plot.magnitude_response(h_0p5, label=r"$\alpha = 0.5$"); \
            sdr.plot.magnitude_response(h_0p9, label=r"$\alpha = 0.9$")

        The span of the filter affects the stopband attenuation. A longer span results in greater stopband
        attenuation and lower sidelobes.

        .. ipython:: python

            h_4 = sdr.raised_cosine(0.1, 4, 10); \
            h_8 = sdr.raised_cosine(0.1, 8, 10); \
            h_16 = sdr.raised_cosine(0.1, 16, 10)

            @savefig sdr_raised_cosine_3.svg
            plt.figure(); \
            sdr.plot.impulse_response(h_4, label="span = 4"); \
            sdr.plot.impulse_response(h_8, label="span = 8"); \
            sdr.plot.impulse_response(h_16, label="span = 16")

            @savefig sdr_raised_cosine_4.svg
            plt.figure(); \
            sdr.plot.magnitude_response(h_4, label="span = 4"); \
            sdr.plot.magnitude_response(h_8, label="span = 8"); \
            sdr.plot.magnitude_response(h_16, label="span = 16")

        See the :ref:`pulse-shapes` example.

    Group:
        modulation-pulse-shaping
    """
    verify_scalar(alpha, float=True, inclusive_min=0, inclusive_max=1)
    verify_scalar(span, int=True, positive=True)
    verify_scalar(sps, int=True, inclusive_min=2)
    verify_scalar(span * sps, even=True)
    verify_literal(norm, ["power", "energy", "passband"])

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
) -> npt.NDArray[np.float64]:
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

    Notes:
        The square root raised cosine pulse shape has a transfer function of

        $$
        H(f) =
        \begin{cases}
        \displaystyle \sqrt{T_{sym}},
        & \displaystyle 0 \le \left| f \right| \le \frac{1 - \alpha}{2 T_{sym}} \\
        \displaystyle \sqrt{\frac{T_{sym}}{2} \left[1 + \cos\left(\frac{\pi T_{sym}}{\alpha}\left(\left| f \right| - \frac{1 - \alpha}{2 T_{sym}}\right)\right)\right]},
        & \displaystyle \frac{1 - \alpha}{2 T_{sym}} \le \left| f \right| \le \frac{1 + \alpha}{2 T_{sym}} \\
        0,
        & \displaystyle \left| f \right| \ge \frac{1 + \alpha}{2 T_{sym}}
        \end{cases}
        $$

        The impulse response is defined as

        $$
        h(t) =
        \frac{1}{\sqrt{T_{sym}}}
        \frac
        {\displaystyle \sin\left(\frac{\pi (1 - \alpha) t}{T_{sym}}\right) + \frac{4 \alpha t}{T_{sym}}\cos\left( \frac{\pi (1 + \alpha) t}{T_{sym}}\right)}
        {\displaystyle \frac{\pi t}{T_{sym}} \left[1 - \left(\frac{4 \alpha t}{T_{sym}} \right)^2 \right]}
        $$

    References:
        - Michael Rice, *Digital Communications: A Discrete Time Approach*, Appendix A.

    Examples:
        The excess bandwidth $\alpha$ controls bandwidth of the filter. A smaller $\alpha$ results in a
        narrower bandwidth at the expense of higher sidelobes.

        .. ipython:: python

            h_0p1 = sdr.root_raised_cosine(0.1, 8, 10); \
            h_0p5 = sdr.root_raised_cosine(0.5, 8, 10); \
            h_0p9 = sdr.root_raised_cosine(0.9, 8, 10)

            @savefig sdr_root_raised_cosine_1.svg
            plt.figure(); \
            sdr.plot.impulse_response(h_0p1, label=r"$\alpha = 0.1$"); \
            sdr.plot.impulse_response(h_0p5, label=r"$\alpha = 0.5$"); \
            sdr.plot.impulse_response(h_0p9, label=r"$\alpha = 0.9$")

            @savefig sdr_root_raised_cosine_2.svg
            plt.figure(); \
            sdr.plot.magnitude_response(h_0p1, label=r"$\alpha = 0.1$"); \
            sdr.plot.magnitude_response(h_0p5, label=r"$\alpha = 0.5$"); \
            sdr.plot.magnitude_response(h_0p9, label=r"$\alpha = 0.9$")

        The span of the filter affects the stopband attenuation. A longer span results in greater stopband
        attenuation and lower sidelobes.

        .. ipython:: python

            h_4 = sdr.root_raised_cosine(0.1, 4, 10); \
            h_8 = sdr.root_raised_cosine(0.1, 8, 10); \
            h_16 = sdr.root_raised_cosine(0.1, 16, 10)

            @savefig sdr_root_raised_cosine_3.svg
            plt.figure(); \
            sdr.plot.impulse_response(h_4, label="span = 4"); \
            sdr.plot.impulse_response(h_8, label="span = 8"); \
            sdr.plot.impulse_response(h_16, label="span = 16")

            @savefig sdr_root_raised_cosine_4.svg
            plt.figure(); \
            sdr.plot.magnitude_response(h_4, label="span = 4"); \
            sdr.plot.magnitude_response(h_8, label="span = 8"); \
            sdr.plot.magnitude_response(h_16, label="span = 16")

        See the :ref:`pulse-shapes` example.

    Group:
        modulation-pulse-shaping
    """
    verify_scalar(alpha, float=True, inclusive_min=0, inclusive_max=1)
    verify_scalar(span, int=True, positive=True)
    verify_scalar(sps, int=True, inclusive_min=2)
    verify_scalar(span * sps, even=True)
    verify_literal(norm, ["power", "energy", "passband"])

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


def _normalize(h: npt.NDArray[np.float64], norm: Literal["power", "energy", "passband"]) -> npt.NDArray[np.float64]:
    if norm == "power":
        h /= np.sqrt(np.max(np.abs(h) ** 2))
    elif norm == "energy":
        h /= np.sqrt(np.sum(np.abs(h) ** 2))
    elif norm == "passband":
        h /= np.sum(h)

    return h
