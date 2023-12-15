"""
A module for specific filter applications.
"""
from __future__ import annotations

import numpy as np
import scipy.signal
from typing_extensions import Literal

from .._helper import export
from ._fir import FIR
from ._iir import IIR


@export
class MovingAverage(FIR):
    r"""
    Implements a moving average FIR filter.

    Notes:
        A discrete-time moving average with length $L$ is a FIR filter with impulse response

        $$h[n] = \frac{1}{L}, \quad 0 \le n \le L - 1 .$$

    Examples:
        Create a FIR moving average filter and a IIR leaky integrator filter.

        .. ipython:: python

            fir = sdr.MovingAverage(30)
            iir = sdr.LeakyIntegrator(1 - 2 / 30)

        Compare the step responses.

        .. ipython:: python

            @savefig sdr_MovingAverage_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.step_response(fir, N=100, label="Moving Average"); \
            sdr.plot.step_response(iir, N=100, label="Leaky Integrator");

        Compare the magnitude responses.

        .. ipython:: python

            @savefig sdr_MovingAverage_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(fir, label="Moving Average"); \
            sdr.plot.magnitude_response(iir, label="Leaky Integrator");

        Compare the output of the two filters to a Gaussian random process.

        .. ipython:: python

            x = np.random.randn(1_000) + 2.0; \
            y_fir = fir(x); \
            y_iir = iir(x)

            @savefig sdr_MovingAverage_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(y_fir, label="Moving Average"); \
            sdr.plot.time_domain(y_iir, label="Leaky Integrator");

    Group:
        dsp-fir-filtering
    """

    def __init__(self, length: int, streaming: bool = False):
        """
        Creates a moving average FIR filter.

        Arguments:
            length: The length of the moving average filter $L$.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~sdr.MovingAverage.__call__()`.

        Examples:
            See the :ref:`fir-filters` example.
        """
        if not isinstance(length, int):
            raise TypeError(f"Argument 'length' must be an integer, not {type(length).__name__}.")
        if not length > 1:
            raise ValueError(f"Argument 'length' must be greater than 1, not {length}.")

        h = np.ones(length) / length

        super().__init__(h, streaming=streaming)


@export
class Differentiator(FIR):
    r"""
    Implements a differentiator FIR filter.

    Notes:
        A discrete-time differentiator is a FIR filter with impulse response

        $$h[n] = \frac{(-1)^n}{n} \cdot h_{win}[n], \quad -\frac{N}{2} \le n \le \frac{N}{2} .$$

        The truncated impulse response is multiplied by the windowing function $h_{win}[n]$.

    References:
        - Michael Rice, *Digital Communications: A Discrete Time Approach*, Section 3.3.3.

    Examples:
        Create a differentiator FIR filter.

        .. ipython:: python

            fir = sdr.Differentiator()

        Differentiate a Gaussian pulse.

        .. ipython:: python

            x = sdr.gaussian(0.3, 5, 10); \
            y = fir(x, "same")

            @savefig sdr_Differentiator_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, label="Input"); \
            sdr.plot.time_domain(y, label="Derivative"); \
            plt.title("Discrete-time differentiation of a Gaussian pulse"); \
            plt.tight_layout();

        Differentiate a raised cosine pulse.

        .. ipython:: python

            x = sdr.root_raised_cosine(0.1, 8, 10); \
            y = fir(x, "same")

            @savefig sdr_Differentiator_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, label="Input"); \
            sdr.plot.time_domain(y, label="Derivative"); \
            plt.title("Discrete-time differentiation of a raised cosine pulse"); \
            plt.tight_layout();

        Plot the frequency response across filter order.

        .. ipython:: python

            fir_2 = sdr.Differentiator(2); \
            fir_6 = sdr.Differentiator(6); \
            fir_10 = sdr.Differentiator(10); \
            fir_20 = sdr.Differentiator(20); \
            fir_40 = sdr.Differentiator(40); \
            fir_80 = sdr.Differentiator(80)

            @savefig sdr_Differentiator_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(fir_2, y_axis="linear", label="N=2"); \
            sdr.plot.magnitude_response(fir_6, y_axis="linear", label="N=6"); \
            sdr.plot.magnitude_response(fir_10, y_axis="linear", label="N=10"); \
            sdr.plot.magnitude_response(fir_20, y_axis="linear", label="N=20"); \
            sdr.plot.magnitude_response(fir_40, y_axis="linear", label="N=40"); \
            sdr.plot.magnitude_response(fir_80, y_axis="linear", label="N=80"); \
            f = np.linspace(0, 0.5, 100); \
            plt.plot(f, np.abs(2 * np.pi * f)**2, color="k", linestyle="--", label="Theory"); \
            plt.legend(); \
            plt.title("Magnitude response of differentiator FIR filters"); \
            plt.tight_layout();

    Group:
        dsp-fir-filtering
    """

    def __init__(self, order: int = 20, window: str | float | tuple | None = "blackman", streaming: bool = False):
        """
        Creates a differentiator FIR filter.

        Arguments:
            order: The order of the FIR differentiator $N$. The filter length is $N + 1$.
                Increasing the filter order increases the bandwidth of the differentiator.
            window: The SciPy window definition. See :func:`scipy.signal.windows.get_window` for details.
                If `None`, no window is applied.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~sdr.Differentiator.__call__()`.

        Examples:
            See the :ref:`fir-filters` example.
        """
        if not isinstance(order, int):
            raise TypeError("Argument 'order' must be an integer, not {type(order).__name__}.")
        if not order > 0:
            raise ValueError(f"Argument 'order' must be positive, not {order}.")
        if not order % 2 == 0:
            raise ValueError(f"Argument 'order' must be even, not {order}.")

        n = np.arange(-order // 2, order // 2 + 1)  # Sample index centered about 0
        with np.errstate(divide="ignore"):
            h = (-1.0) ** n / n  # Impulse response
        h[order // 2] = 0

        if window is not None:
            h_win = scipy.signal.windows.get_window(window, order + 1, fftbins=False)
            h *= h_win

        super().__init__(h, streaming=streaming)

    # TODO: Use np.diff() if it is faster


@export
class Integrator(IIR):
    r"""
    Implements an integrator IIR filter.

    Notes:
        A discrete-time integrator is an IIR filter that continuously accumulates the input signal.
        Accordingly, it has infinite gain at DC.

        The backward integrator is defined by:

        $$y[n] = y[n-1] + x[n-1]$$
        $$H(z) = \frac{z^{-1}}{1 - z^{-1}}$$

        The trapezoidal integrator is defined by:

        $$y[n] = y[n-1] + \frac{1}{2}x[n] + \frac{1}{2}x[n-1]$$
        $$H(z) = \frac{1}{2} \frac{1 + z^{-1}}{1 - z^{-1}}$$

        The forward integrator is defined by:

        $$y[n] = y[n-1] + x[n]$$
        $$H(z) = \frac{1}{1 - z^{-1}}$$

    Examples:
        Create integrating IIR filters.

        .. ipython:: python

            iir_back = sdr.Integrator("backward"); \
            iir_trap = sdr.Integrator("trapezoidal"); \
            iir_forw = sdr.Integrator("forward")

        Integrate a Gaussian pulse.

        .. ipython:: python

            x = sdr.gaussian(0.3, 5, 10); \
            y_back = iir_back(x); \
            y_trap = iir_trap(x); \
            y_forw = iir_forw(x)

            @savefig sdr_Integrator_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, label="Input"); \
            sdr.plot.time_domain(y_back, label="Integral (backward)"); \
            sdr.plot.time_domain(y_trap, label="Integral (trapezoidal)"); \
            sdr.plot.time_domain(y_forw, label="Integral (forward)"); \
            plt.title("Discrete-time integration of a Gaussian pulse"); \
            plt.tight_layout();

        Integrate a raised cosine pulse.

        .. ipython:: python

            x = sdr.root_raised_cosine(0.1, 8, 10); \
            y_back = iir_back(x); \
            y_trap = iir_trap(x); \
            y_forw = iir_forw(x)

            @savefig sdr_Integrator_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, label="Input"); \
            sdr.plot.time_domain(y_back, label="Integral (backward)"); \
            sdr.plot.time_domain(y_trap, label="Integral (trapezoidal)"); \
            sdr.plot.time_domain(y_forw, label="Integral (forward)"); \
            plt.title("Discrete-time integration of a raised cosine pulse"); \
            plt.tight_layout();

        Plot the frequency responses.

        .. ipython:: python

            @savefig sdr_Integrator_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(iir_back, label="Backward"); \
            sdr.plot.magnitude_response(iir_trap, label="Trapezoidal"); \
            sdr.plot.magnitude_response(iir_forw, label="Forward"); \
            f = np.linspace(0, 0.5, 100); \
            plt.plot(f, sdr.db(np.abs(1/(2 * np.pi * f))**2), color="k", linestyle="--", label="Theory"); \
            plt.legend(); \
            plt.title("Magnitude response of integrating IIR filters"); \
            plt.tight_layout();

    Group:
        dsp-iir-filtering
    """

    def __init__(self, method: Literal["backward", "trapezoidal", "forward"] = "trapezoidal", streaming: bool = False):
        """
        Creates an integrating IIR filter.

        Arguments:
            method: The integration method.

                - `"backward"`: Rectangular integration with height $x[n-1]$.
                - `"trapezoidal"`: Trapezoidal integration with heights $x[n-1]$ and $x[n]$.
                - `"forward"`: Rectangular integration with height $x[n]$.

            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs and outputs are
                preserved between calls to :meth:`~Integrator.__call__()`.

        Examples:
            See the :ref:`iir-filters` example.
        """
        if method == "backward":
            super().__init__([0, 1], [1, -1], streaming=streaming)
        elif method == "forward":
            super().__init__([1], [1, -1], streaming=streaming)
        elif method == "trapezoidal":
            super().__init__([0.5, 0.5], [1, -1], streaming=streaming)
        else:
            raise ValueError(f"Argument 'method' must be 'backward', 'forward', or 'trapezoidal', not {method!r}.")

    # TODO: Use np.cumsum() if it is faster


@export
class LeakyIntegrator(IIR):
    r"""
    Implements a leaky integrator IIR filter.

    Notes:
        A discrete-time leaky integrator is an IIR filter that approximates an FIR moving average.
        The previous output is remembered with the leaky factor $\alpha$ and the new input is scaled with $1 - \alpha$.

        The difference equation is

        $$y[n] = \alpha \cdot y[n-1] + (1 - \alpha) \cdot x[n] .$$

        The transfer functions is

        $$H(z) = \frac{1 - \alpha}{1 - \alpha z^{-1}} .$$

        .. code-block:: text
            :caption: IIR Integrator Block Diagram

                  1 - alpha
            x[n] ----------->@---------------+--> y[n]
                             ^               |
                       alpha |   +------+    |
                             +---| z^-1 |<---+
                                 +------+

    Examples:
        Compare leaky integrators with various leaky factors.

        .. ipython:: python

            iir_0p99 = sdr.LeakyIntegrator(0.99); \
            iir_0p9 = sdr.LeakyIntegrator(0.9); \
            iir_0p75 = sdr.LeakyIntegrator(0.75); \
            iir_0p5 = sdr.LeakyIntegrator(0.5)

        Plot the impulse responses.

        .. ipython:: python

            @savefig sdr_LeakyIntegrator_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(iir_0p99, label=r"$\alpha=0.99$"); \
            sdr.plot.impulse_response(iir_0p9, label=r"$\alpha=0.9$"); \
            sdr.plot.impulse_response(iir_0p75, label=r"$\alpha=0.75$"); \
            sdr.plot.impulse_response(iir_0p5, label=r"$\alpha=0.5$"); \
            plt.xlim(0, 50); \
            plt.tight_layout();

        Plot the step responses.

        .. ipython:: python

            @savefig sdr_LeakyIntegrator_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.step_response(iir_0p99, label=r"$\alpha=0.99$"); \
            sdr.plot.step_response(iir_0p9, label=r"$\alpha=0.9$"); \
            sdr.plot.step_response(iir_0p75, label=r"$\alpha=0.75$"); \
            sdr.plot.step_response(iir_0p5, label=r"$\alpha=0.5$"); \
            plt.xlim(0, 50); \
            plt.tight_layout();

        Plot the frequency responses.

        .. ipython:: python

            @savefig sdr_LeakyIntegrator_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(iir_0p99, label=r"$\alpha=0.99$"); \
            sdr.plot.magnitude_response(iir_0p9, label=r"$\alpha=0.9$"); \
            sdr.plot.magnitude_response(iir_0p75, label=r"$\alpha=0.75$"); \
            sdr.plot.magnitude_response(iir_0p5, label=r"$\alpha=0.5$"); \
            plt.tight_layout();

        Integrate a raised cosine pulse.

        .. ipython:: python

            x = sdr.root_raised_cosine(0.1, 8, 10); \
            y_0p99 = iir_0p99(x); \
            y_0p9 = iir_0p9(x); \
            y_0p75 = iir_0p75(x); \
            y_0p5 = iir_0p5(x)

            @savefig sdr_LeakyIntegrator_4.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, label="Input"); \
            sdr.plot.time_domain(y_0p99, label=r"$\alpha=0.99$ Integral"); \
            sdr.plot.time_domain(y_0p9, label=r"$\alpha=0.9$ Integral"); \
            sdr.plot.time_domain(y_0p75, label=r"$\alpha=0.75$ Integral"); \
            sdr.plot.time_domain(y_0p5, label=r"$\alpha=0.5$ Integral"); \
            plt.title("Discrete-time integration of a raised cosine pulse"); \
            plt.tight_layout();

    Group:
        dsp-iir-filtering
    """

    def __init__(self, alpha: float, streaming: bool = False):
        r"""
        Creates a leaky integrator IIR filter.

        Arguments:
            alpha: The leaky factor $\alpha$. An FIR moving average with length $L$ is approximated when
                $\alpha = 1 - 2/L$.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs and outputs are
                preserved between calls to :meth:`~LeakyIntegrator.__call__()`.

        Examples:
            See the :ref:`iir-filters` example.
        """
        if not isinstance(alpha, float):
            raise TypeError(f"Argument 'alpha' must be a float, not {type(alpha).__name__}.")
        if not 0 <= alpha <= 1:
            raise ValueError(f"Argument 'alpha' must be between 0 and 1, not {alpha}.")

        b = [1 - alpha]
        a = [1, -alpha]

        super().__init__(b, a, streaming=streaming)

    # TODO: Use np.cumsum() if it is faster
