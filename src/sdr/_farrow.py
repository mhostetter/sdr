"""
A module containing a Farrow arbitrary resampler.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal

from ._helper import convert_output, export, verify_arraylike, verify_bool, verify_scalar


@export
class FarrowFractionalDelay:
    r"""
    Implements a piecewise polynomial Farrow fractional delay filter.

    The Farrow fractional delay filter uses a piecewise polynomial to approximate a sinc interpolator. The filter is
    implemented as a set of FIR filters, where each filter corresponds to a polynomial order of $\mu$.

    $$y[n] = x((m(k) + \mu(k)) T_s) = x[m(k) + \mu(k)]$$

    References:
        - Michael Rice, *Digital Communications: A Discrete Time Approach*, Section 8.4.2.
        - `Qasim Chaudhari, Fractional Delay Filters Using the Farrow Structure.
          <https://wirelesspi.com/fractional-delay-filters-using-the-farrow-structure/>`_

    Examples:
        Examples.

    Group:
        dsp-arbitrary-resampling
    """

    def __init__(self, order: int = 3, alpha: float = 0.5, streaming: bool = False):
        self._order = verify_scalar(order, int=True, inclusive_min=1, inclusive_max=3)
        self._streaming = verify_bool(streaming)
        self._state: npt.NDArray  # FIR filter state. Will be updated in reset().

        if self.order == 1:
            # Equation 8.63
            self._delay = 1
            self._taps = np.array(
                [
                    [1, -1],  # b1(i) = h1[-1, 0] for mu^1
                    [0, 1],  # b0(i) = h1[-1, 0] for mu^0
                ]
            )
        elif self.order == 2:
            # Table 8.4.1
            self._delay = 2
            self._taps = np.array(
                [
                    [alpha, -alpha, -alpha, alpha],  # b2(i) = h2[-2, -1, 0, 1] for mu^2
                    [-alpha, 1 + alpha, alpha - 1, -alpha],  # b1(i) = h2[-2, -1, 0, 1] for mu^3
                    [0, 0, 1, 0],  # b0(i) = h2[-2, -1, 0, 1] for mu^0
                ]
            )
        elif self.order == 3:
            # Table 8.4.2
            self._delay = 2
            self._taps = np.array(
                [
                    [1 / 6, -1 / 2, 1 / 2, -1 / 6],  # b3(i) = h3[-2, -1, 0, 1] for mu^3
                    [0, 1 / 2, -1, 1 / 2],  # b2(i) = h3[-2, -1, 0, 1] for mu^2
                    [-1 / 6, 1, -1 / 2, -1 / 3],  # b1(i) = h3[-2, -1, 0, 1] for mu^1
                    [0, 0, 1, 0],  # b0(i) = h3[-2, -1, 0, 1] for mu^0
                ]
            )

        self.reset()

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(
        self,
        x: npt.ArrayLike,
        m: npt.ArrayLike | None = None,
        mu: npt.ArrayLike | None = None,
    ) -> npt.NDArray:
        r"""
        Applies the fractional delay $\mu(k)$ to the input signal $x[n]$ at the given basepoint sample indices $m(k)$.

        $$y[n] = x((m(k) + \mu(k)) T_s) = x[m(k) + \mu(k)]$$

        Arguments:
            x: The input signal $x[n] = x(n T_s)$.
            m: The basepoint sample indices $m(k)$, which are the integer sample indices of the input signal.
            mu: The fractional sample delay $0 \le \mu(k) \le 1$, which is the fractional sample delay of the
                input signal at input sample $m(k)$.

        Returns:
            The resampled signal $y[k]$.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        x = verify_arraylike(x, complex=True, atleast_1d=True, ndim=1)

        if m is None:
            m = np.arange(0, 0 + x.size)
        else:
            m = verify_arraylike(
                m, int=True, inclusive_min=0, exclusive_max=self.delay + x.size, atleast_1d=True, ndim=1
            )

        if mu is None:
            mu = np.zeros_like(x, dtype=float)
        else:
            mu = verify_arraylike(mu, float=True, inclusive_min=0, inclusive_max=1, atleast_1d=True, ndim=1)

        # m and mu can be scalars or arrays. If they are arrays, they must be the same size.
        m, mu = np.broadcast_arrays(m, mu)

        if self.streaming:
            # Prepend previous inputs from last streaming call
            x_pad = np.concatenate((self._state, x))

            # Compute the FIR filter outputs for the entire input signal
            ys = []
            for i in range(self.order + 1):
                yi = scipy.signal.convolve(x_pad, self._taps[i, :], mode="full")
                ys.append(yi)

            # Offset the basepoint sample indices by the number of historical samples prepended
            m += self._state.size

            # Store the previous inputs for the next call to __call__()
            self._state = x_pad[-(self._taps.shape[1] - 1) :]
        else:
            # Compute the FIR filter outputs for the entire input signal
            ys = []
            for i in range(self.order + 1):
                yi = scipy.signal.convolve(x, self._taps[i, :], mode="full")
                ys.append(yi)

        # Interpolate the output samples using the Horner method
        # The definition of mu looks back from self.delay. So mu is technically and advance. We instead consider
        # mu as a delay from self.delay - 1. Since the filters are symmetric about mu = 0.5, we can do this.
        y = ys[0][m + self.delay - 1]
        for i in range(1, self.order + 1):
            y *= 1 - mu
            y += ys[i][m + self.delay - 1]

        return convert_output(y)

    ##############################################################################
    # Streaming mode
    ##############################################################################

    def reset(self, state: npt.ArrayLike | None = None):
        """
        Resets the filter state.

        Arguments:
            state: The filter state to reset to. The state vector should equal the previous `self.taps.shape[1] - 1`
                inputs. If `None`, the filter state will be set to `[]`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.

        Group:
            Streaming mode only
        """
        if state is None:
            self._state = np.array([])
        else:
            self._state = verify_arraylike(state, complex=True, size=self._taps.shape[1] - 1)

    @property
    def streaming(self) -> bool:
        """
        Indicates whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :meth:`~FarrowResampler.__call__()`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.

        Group:
            Streaming mode only
        """
        return self._streaming

    @property
    def state(self) -> npt.NDArray:
        """
        The filter state consisting of the previous `self.taps.shape[1] - 1` inputs.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.

        Group:
            Streaming mode only
        """
        return self._state

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def order(self) -> int:
        """
        The order of the piecewise polynomial.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._order

    @property
    def taps(self) -> npt.NDArray:
        r"""
        The Farrow filter taps.

        The taps are in the form of a 2D array, where each row corresponds to a polynomial order of $\mu$ and
        each column corresponds to a tap. The taps are ordered from the highest order polynomial to the
        lowest order polynomial.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._taps

    @property
    def delay(self) -> int:
        r"""
        The delay $d$ of the Farrow FIR filters in samples.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._delay


@export
class FarrowResampler:
    r"""
    Implements a piecewise polynomial Farrow arbitrary resampler.

    References:
        - Michael Rice, *Digital Communications: A Discrete Time Approach*, Section 8.4.2.
        - `Qasim Chaudhari, Fractional Delay Filters Using the Farrow Structure.
          <https://wirelesspi.com/fractional-delay-filters-using-the-farrow-structure/>`_

    Examples:
        Create a sine wave with angular frequency $\omega = 2 \pi / 5.179$. Interpolate the signal by
        $r = \pi$ using Farrow piecewise polynomial Farrow resamplers.

        .. ipython:: python

            x = np.cos(2 * np.pi / 5.179 * np.arange(11))
            rate = np.pi

        Create a linear Farrow piecewise polynomial interpolator.

        .. ipython:: python

            farrow1 = sdr.FarrowResampler(1)
            y1 = farrow1(x, rate)

            @savefig sdr_FarrowResampler_1.png
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y1, sample_rate=rate, marker=".", label="Output"); \
            plt.title("Linear Farrow Resampler");

        Create a quadratic Farrow piecewise polynomial interpolator.

        .. ipython:: python

            farrow2 = sdr.FarrowResampler(2)
            y2 = farrow2(x, rate)

            @savefig sdr_FarrowResampler_2.png
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y2, sample_rate=rate, marker=".", label="Output"); \
            plt.title("Quadratic Farrow Resampler");

        Create a cubic Farrow piecewise polynomial interpolator.

        .. ipython:: python

            farrow3 = sdr.FarrowResampler(3)
            y3 = farrow3(x, rate)

            @savefig sdr_FarrowResampler_3.png
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y3, sample_rate=rate, marker=".", label="Output"); \
            plt.title("Cubic Farrow Resampler");

        Create a quartic Farrow piecewise polynomial interpolator.

        .. ipython:: python

            farrow4 = sdr.FarrowResampler(4)
            y4 = farrow4(x, rate)

            @savefig sdr_FarrowResampler_4.png
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y4, sample_rate=rate, marker=".", label="Output"); \
            plt.title("Quartic Farrow Resampler");

        Compare the outputs of the Farrow resamplers with varying polynomial order.

        .. ipython:: python

            @savefig sdr_FarrowResampler_5.png
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y1, sample_rate=rate, marker=".", label="Linear"); \
            sdr.plot.time_domain(y2, sample_rate=rate, marker=".", label="Quadratic"); \
            sdr.plot.time_domain(y3, sample_rate=rate, marker=".", label="Cubic"); \
            sdr.plot.time_domain(y4, sample_rate=rate, marker=".", label="Quartic"); \
            plt.xlim(1.5, 3.5); \
            plt.ylim(-1.0, -0.2); \
            plt.title("Comparison of Farrow Resamplers");

        Run a Farrow resampler with quartic polynomial order in streaming mode.

        .. ipython:: python

            x = np.cos(2 * np.pi / 5.179 * np.arange(40))
            farrow4 = sdr.FarrowResampler(4, streaming=True)

            y1 = farrow4(x[0:10], rate); \
            y2 = farrow4(x[10:20], rate); \
            y3 = farrow4(x[20:30], rate); \
            y4 = farrow4(x[30:40], rate); \
            y5 = farrow4.flush(rate); \
            y = np.concatenate((y1, y2, y3, y4, y5))

            @savefig sdr_FarrowResampler_6.png
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=rate, offset=-farrow4.delay, marker=".", label="Quartic concatenated"); \
            plt.title("Quartic Farrow Resampler Concatenated Outputs");

        See the :ref:`farrow-arbitrary-resampler` example.

    Group:
        dsp-arbitrary-resampling
    """

    def __init__(self, order: int = 3, streaming: bool = False):
        """
        Creates a new Farrow arbitrary resampler.

        Arguments:
            order: The order of the piecewise polynomial, which must be between 1 and 4.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~FarrowResampler.__call__()`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        self._order = verify_scalar(order, int=True, inclusive_min=1, inclusive_max=4)
        self._streaming = verify_bool(streaming)
        self._state: npt.NDArray  # FIR filter state. Will be updated in reset().
        self._mu_next: float  # The next fractional sample delay value

        if self.order == 1:
            self._delay = 1
            self._taps = np.array(
                [
                    [-1, 1],
                    [1, 0],
                ]
            ).T
        elif self.order == 2:
            self._delay = 2
            self._taps = np.array(
                [
                    [1 / 2, -1 / 2, 0],
                    [-1, 0, 1],
                    [1 / 2, 1 / 2, 0],
                ]
            ).T
        elif self.order == 3:
            self._delay = 2
            self._taps = np.array(
                [
                    [-1 / 6, 1 / 2, -1 / 3, 0],
                    [1 / 2, -1, -1 / 2, 1],
                    [-1 / 2, 1 / 2, 1, 0],
                    [1 / 6, 0, -1 / 6, 0],
                ]
            ).T
        elif self.order == 4:
            self._delay = 3
            self._taps = np.array(
                [
                    [1 / 24, -1 / 12, -1 / 24, 1 / 12, 0],
                    [-1 / 6, 1 / 6, 2 / 3, -2 / 3, 0],
                    [1 / 4, 0, -5 / 4, 0, 1],
                    [-1 / 6, -1 / 6, 2 / 3, 2 / 3, 0],
                    [1 / 24, 1 / 12, -1 / 24, -1 / 12, 0],
                ]
            ).T

        self.reset()

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike, rate: float) -> npt.NDArray:
        r"""
        Resamples the input signal $x[n]$ by the given arbitrary rate $r$.

        Arguments:
            x: The input signal $x[n] = x(n T_s)$.
            rate: The resampling rate $r$.

        Returns:
            The resampled signal $y[n] = x(n T_s / r)$.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        x = verify_arraylike(x, complex=True, atleast_1d=True, ndim=1)
        verify_scalar(rate, float=True, positive=True)

        if self.streaming:
            # Prepend previous inputs from last streaming call
            x_pad = np.concatenate((self._state, x))

            # Compute the FIR filter outputs for the entire input signal
            ys = []
            for i in range(self.order + 1):
                yi = scipy.signal.convolve(x_pad, self._taps[i, :], mode="full")
                ys.append(yi)

            # Compute the fractional sample indices for each output sample. We want to step from mu_next to
            # y0.size in steps of 1 / rate.
            frac_idxs = np.arange(
                self._mu_next,
                x_pad.size,
                1 / rate,
            )

            # Store the previous inputs and next fractional sample index for the next call to __call__()
            self._state = x_pad[-(self._taps.shape[1] - 1) :]
            self._mu_next = (frac_idxs[-1] + 1 / rate) - x_pad.size + (self._taps.shape[1] - 1)
        else:
            # Compute the FIR filter outputs for the entire input signal
            ys = []
            for i in range(self.order + 1):
                yi = scipy.signal.convolve(x, self._taps[i, :], mode="full")
                ys.append(yi)

            # Compute the fractional sample indices for each output sample
            frac_idxs = np.arange(
                self._delay,  # Account for filter delay
                self._delay + x.size,
                1 / rate,
            )

        # Convert the fractional indices to integer indices and fractional indices
        int_idxs = (frac_idxs // 1.0).astype(int)
        mu = frac_idxs - int_idxs
        mu *= -1  # TODO: Why is this the case?

        # Interpolate the output samples using the Horner method
        y = ys[0][int_idxs]
        for i in range(1, self.order + 1):
            y += mu * y + ys[i][int_idxs]

        return convert_output(y)

    ##############################################################################
    # Streaming mode
    ##############################################################################

    def reset(self, state: npt.ArrayLike | None = None):
        """
        Resets the filter state and fractional sample index.

        Arguments:
            state: The filter state to reset to. The state vector should equal the previous three
                inputs. If `None`, the filter state will be reset to zero.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.

        Group:
            Streaming mode only
        """
        if state is None:
            self._state = np.array([])
        else:
            self._state = verify_arraylike(state, complex=True, size=self._taps.shape[1] - 1)

        # Initial fractional sample delay accounts for filter delay
        self._mu_next = 0

    def flush(self, rate: float) -> npt.NDArray:
        """
        Flushes the filter state by passing zeros through the filter. Only useful when using streaming mode.

        Arguments:
            rate: The resampling rate $r$.

        Returns:
            The remaining resampled signal $y[n]$.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.

        Group:
            Streaming mode only
        """
        verify_scalar(rate, float=True, positive=True)

        x = np.zeros_like(self.state)
        y = self(x, rate)

        return y

    @property
    def streaming(self) -> bool:
        """
        Indicates whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :meth:`~FarrowResampler.__call__()`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.

        Group:
            Streaming mode only
        """
        return self._streaming

    @property
    def state(self) -> npt.NDArray:
        """
        The filter state consisting of the previous $N$ inputs.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.

        Group:
            Streaming mode only
        """
        return self._state

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def order(self) -> int:
        """
        The order of the piecewise polynomial.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._order

    @property
    def taps(self) -> npt.NDArray:
        """
        The Farrow filter taps.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._taps

    @property
    def delay(self) -> int:
        r"""
        The delay $d$ of the Farrow FIR filters in samples.

        Output sample $d \cdot r$, corresponds to the first input sample, where $r$ is the current resampling rate.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._delay
