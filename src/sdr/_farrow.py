"""
A module containing a Farrow arbitrary resampler.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.signal
from typing_extensions import Literal

from ._helper import convert_output, export, verify_arraylike, verify_bool, verify_literal, verify_scalar


@export
class FarrowFractionalDelay:
    r"""
    Implements a piecewise polynomial Farrow fractional delay filter.

    Consider the series of input samples $x[n-D], \ldots, x[n], \ldots, x[n+p-D]$. The goal is to interpolate the
    value of $x(t)$ at the time $t = n + \mu$. The Lagrange polynomial interpolation is computed as

    $$x[n + \mu] = \sum_{k=0}^{p} x[n + k - D] \cdot \ell_k(\mu) ,$$

    where $D$ is the delay offset (usually so the center tap is at zero) and $\ell_k(\mu)$ is the Lagrange basis
    polynomial corresponding to tap $k$. Note that $\ell_k(\mu)$ is independent of the data and only depends on the
    fractional shift $\mu$.

    The $k$-th Lagrange basis polynomial is defined as

    $$\ell_k(\mu) = \prod_{\substack{j=0 \\ j \ne k}}^{p} \frac{\mu - j}{k - j} ,$$

    which is a polynomial of degree $p$ in $\mu$. The Lagrange basis polynomials are 1 at $\mu = k$ and 0 at
    $\mu = j$ for $j \ne k$.

    It is often desirable to change $\mu$ on the fly, which means that the Lagrange basis polynomials must be
    recomputed for each new value of $\mu$. Instead, the Farrow structure selects the coefficients of each
    Lagrange basis polynomial for a given degree of $\mu$ and forms it into a correlator

    $$h_m[n] = \{\ell_{0,m}, \ell_{1,m}, \ldots, \ell_{p,m}\} ,$$

    where $h_m[n]$ is the $m$-th correlator and $\ell_{k,m}$ is the coefficient of $\mu^m$ in the $k$-th Lagrange
    basis polynomial.

    The output of the $m$-th correlator is

    $$z_m[n] = x[n] \star h_m[n] = \sum_{k=0}^{p} x[n + k - D] \cdot \ell_{k,m} .$$

    The Farrow structure allows the interpolation calculation to be reorganized as

    $$x[n + \mu] = \sum_{k=0}^{p} x[n + k - D] \cdot \ell_k(\mu) = \sum_{k=0}^{p} z_m[n] \cdot \mu^k .$$

    which can be efficiently computed using Horner's method.

    $$x[n + \mu] = \sum_{k=0}^{p} z_m[k] \cdot \mu^k = z_0[n] + \mu \cdot (z_1[n] + \mu \cdot (z_2[n] + \ldots)) .$$

    References:
        - Michael Rice, *Digital Communications: A Discrete Time Approach*, Section 8.4.2.
        - `Qasim Chaudhari, Fractional Delay Filters Using the Farrow Structure.
          <https://wirelesspi.com/fractional-delay-filters-using-the-farrow-structure/>`_

    Examples:
        Plot the 3rd order Lagrange basis polynomials. Notice that each polynomial is 1 and 0 at the corresponding
        indices.

        .. ipython:: python

            farrow = sdr.FarrowFractionalDelay(3)
            mu = np.linspace(0, farrow.order, 1000) - farrow.lookahead

            plt.figure();
            for i in range(0, farrow.order + 1):
                y = np.poly1d(farrow.lagrange_polys[i])(mu)
                sdr.plot.time_domain(mu, y, label=rf"$\ell_{i}(\mu)$");
            @savefig sdr_FarrowFractionalDelay_1.svg
            plt.xlabel("Fractional sample delay, $\mu$"); \
            plt.title("3rd order Lagrange basis polynomials");

        Suppose a signal $x[n]$ is to be interpolated. The interpolated signal $y[n]$ is calculated by evaluating
        the Lagrange interpolating polynomial at the fractional sample delay $\mu$ and scaling by the input signal
        $x[n]$.

        .. ipython:: python

            x = np.array([1, 3, 2, 0])

            y = 0;
            for i in range(0, farrow.order + 1):
                y += x[i] * np.poly1d(farrow.lagrange_polys[i])(mu)

            @savefig sdr_FarrowFractionalDelay_2.svg
            plt.figure(); \
            sdr.plot.time_domain(x, offset=-farrow.lookahead, marker=".", linestyle="none", label="Input"); \
            sdr.plot.time_domain(mu, y, label="Interpolated"); \
            plt.title("3rd order Lagrange interpolation");

        Compare fractional sample delays for various order Farrow fractional delay filters.

        .. ipython:: python

            sps = 6; \
            span = 4; \
            x = sdr.root_raised_cosine(0.5, span, sps, norm="power")

            @savefig sdr_FarrowFractionalDelay_3.svg
            mu = 0.25; \
            plt.figure(); \
            sdr.plot.time_domain(x, marker=".", color="k"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(1)(x, mu=mu), label="Farrow 1"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(2)(x, mu=mu), label="Farrow 2"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(3)(x, mu=mu), label="Farrow 3"); \
            plt.title(f"Fractional advance {mu} samples");

            @savefig sdr_FarrowFractionalDelay_4.svg
            mu = 0.5; \
            plt.figure(); \
            sdr.plot.time_domain(x, marker=".", color="k"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(1)(x, mu=mu), label="Farrow 1"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(2)(x, mu=mu), label="Farrow 2"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(3)(x, mu=mu), label="Farrow 3"); \
            plt.title(f"Fractional advance {mu} samples");

            @savefig sdr_FarrowFractionalDelay_5.svg
            mu = 1; \
            plt.figure(); \
            sdr.plot.time_domain(x, marker=".", color="k"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(1)(x, mu=mu), label="Farrow 1"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(2)(x, mu=mu), label="Farrow 2"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(3)(x, mu=mu), label="Farrow 3"); \
            plt.title(f"Fractional advance {mu} samples");

    Group:
        dsp-arbitrary-resampling
    """

    def __init__(self, order: int, alpha: float = 0.5, streaming: bool = False):
        r"""
        Creates a new Farrow arbitrary fractional delay filter.

        Arguments:
            order: The order $p$ of the Lagrange interpolating polynomial.
            alpha: A free design parameter $0 \le \alpha \le 1$ that controls the shape of a 2nd order filter.
                This ensures that the filter has an even number of taps and is linear phase. The default value
                is $\alpha = 0.5$, which is a good compromise between performance and fixed-point computational
                complexity. It was found through simulation that $\alpha = 0.43$ is optimal for BPSK using a
                square root raised cosine filter with 100% excess bandwidth.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~FarrowFractionalDelay.__call__()`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        self._order = verify_scalar(order, int=True, positive=True)
        self._alpha = verify_scalar(alpha, float=True, inclusive_min=0, inclusive_max=1)
        self._streaming = verify_bool(streaming)
        self._state: npt.NDArray  # FIR filter state. Will be updated in reset().

        self._lookahead, self._lagrange_polys, self._taps = _compute_lagrange_basis(self._order)
        self._delay = self._taps.shape[1] - self._lookahead - 1  # The number of samples needed before the current input
        self._n_extra = 1  # Save one extra sample in the state than necessary

        self.reset()

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(
        self,
        x: npt.ArrayLike,
        m: npt.ArrayLike | None = None,
        mu: npt.ArrayLike | None = None,
        mode: Literal["rate", "full"] = "rate",
    ) -> npt.NDArray:
        r"""
        Applies the fractional sample advance $\mu(k)$ to the input signal $x[n]$ at the given basepoint sample
        indices $m(k)$.

        $$y[k] = x((m(k) + \mu(k)) T_s) = x[m(k) + \mu(k)]$$

        Arguments:
            x: The input signal $x[n] = x(n T_s)$.
            m: The basepoint sample indices $m(k)$, which are the integer sample indices of the input signal.
            mu: The fractional sample indices $0 \le \mu(k) \le 1$, which is the fractional sample advance of the
                input signal at input sample $m(k)$.
            mode: The convolution mode.

                - `"rate"`: The output signal $y[k]$ is aligned with the input signal, such that $y[0] = x[0 + \mu]$.

                  In non-streaming mode, $L - D$ output samples are produced. In streaming mode, the first call returns
                  $L - D$ output samples, where $L$ is the length of the basepoint and fractional sample indices.
                  On subsequent calls, $L$ output samples are produced.

                - `"full"`: The full convolution is performed, and the filter delay $D$ is observed, such that
                  $y[D] = x[0 + \mu]$.

                  In non-streaming mode, $L$ output samples are produced. In streaming mode, each call returns $L$
                  output samples.

        Returns:
            The resampled signal $y[k]$.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        x = verify_arraylike(x, complex=True, atleast_1d=True, ndim=1)
        verify_literal(mode, ["rate", "full"])

        if m is None:
            # Apply mu to each input sample
            m = np.arange(0, x.size)
        else:
            m_min = -self._n_extra
            if mode == "rate":
                m_min -= self.delay
            m = verify_arraylike(m, int=True, inclusive_min=m_min, exclusive_max=x.size, atleast_1d=True, ndim=1)

        if mu is None:
            # If not provided, do not delay the input samples
            mu = np.zeros_like(x, dtype=float)
        else:
            mu = verify_arraylike(mu, float=True, inclusive_min=0, inclusive_max=1, atleast_1d=True, ndim=1)

        # m and mu can be scalars or arrays. If they are arrays, they must be the same size. This NumPy function
        # should error if the sizes are not broadcast-able.
        m, mu = np.broadcast_arrays(m, mu)

        # Prepend the pervious state (zeros initially)
        x_pad = np.concatenate((self._state, x))
        m = np.concatenate((self._m_state, m))
        mu = np.concatenate((self._mu_state, mu))

        z = []
        for i in range(self.order + 1):
            zi = scipy.signal.correlate(x_pad, self._taps[i, :], mode="valid")
            z.append(zi)

        m += self._n_extra
        if mode == "rate":
            m += self.delay

        last_m = x_pad.size - (self._taps.shape[1] - 1)

        if self.streaming:
            # If in streaming mode, we will repeatedly call this function. A valid output correlates with some
            # previous inputs and some future inputs. Once we added the delay, some of the m values will be
            # outside the valid correlation zone. They correspond to correlation outputs with some zeros inputs
            # (from the "full" convolution). We need to only consider the valid m values during this call and
            # save the rest for the next call.

            # Save the m values needed to be processed next call. We also subtract from the m values so that
            # the state is in [-delay, 0) range.
            self._m_state = m[m >= last_m] - x.size - self.delay - self._n_extra
            self._mu_state = mu[m >= last_m]

        # Keep only the valid m values for this call
        mu = mu[m < last_m]  # NOTE: Need to modify m last since it's used in the indexing
        m = m[m < last_m]

        # Interpolate the output samples using the Horner method
        y = z[0][m]
        for i in range(1, self.order + 1):
            y *= mu
            y += z[i][m]

        if self.streaming:
            self._state = x_pad[-(self._taps.shape[1] - 1 + self._n_extra) :]

        return convert_output(y)

    def __repr__(self) -> str:
        return f"sdr.{type(self).__name__}({self.order}, streaming={self.streaming})"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  taps: {self.taps.shape} shape"
        string += f"\n    {np.array2string(self.taps, prefix='    ')}"
        string += f"\n  delay: {self.delay}"
        string += f"\n  lookahead: {self.lookahead}"
        string += f"\n  streaming: {self.streaming}"
        return string

    ##############################################################################
    # Streaming mode
    ##############################################################################

    def reset(self):
        """
        Resets the filter state.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.

        Group:
            Streaming mode only
        """
        self._state = np.zeros(self._taps.shape[1] - 1 + self._n_extra, dtype=float)
        self._m_state = np.zeros(0, dtype=int)
        self._mu_state = np.zeros(0, dtype=float)

    @property
    def streaming(self) -> bool:
        """
        Indicates whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :meth:`~FarrowFractionalDelay.__call__()`.

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
        The order $p$ of the Lagrange interpolating polynomial.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._order

    @property
    def lagrange_polys(self) -> npt.NDArray:
        r"""
        The Lagrange basis polynomials $\ell_k(\mu)$.

        The Lagrange basis polynomials are in the form of a 2D array with rows
        $\{\ell_0(\mu), \ell_1(\mu), \ldots, \ell_p(\mu)\}$ and columns $\{\mu^p, \mu^{p-1}, \ldots, \mu^0\}$.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._lagrange_polys

    @property
    def taps(self) -> npt.NDArray:
        r"""
        The Farrow filter taps.

        The taps are in the form of a 2D array with rows $\{h_0[n], h_1[n], \ldots, h_p[n]\}$, where $h_k[n]$ is the
        $k$-th FIR filter corresponding to $\mu^k$.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._taps

    @property
    def delay(self) -> int:
        r"""
        The delay $D$ of the Farrow FIR filters in samples.

        The delay is only observed when the convolution mode is set to `"full"`. No delay is observed when the
        convolution mode is set to `"rate"`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._delay

    @property
    def lookahead(self) -> int:
        r"""
        The number of samples needed before the current input sample.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._lookahead


@export
class FarrowResampler(FarrowFractionalDelay):
    r"""
    Implements a piecewise polynomial Farrow arbitrary resampler.

    Notes:
        See :class:`~FarrowFractionalDelay` for a detailed description of the Farrow structure and Farrow filter.

        The Farrow resampler is a special implementation of the Farrow filter that is used for arbitrary resampling.
        The resampling rate $r$ is a real-valued number that can be any positive value. The resampling rate
        $r$ is defined as the ratio of the output sample rate to the input sample rate. The resampling rate is used
        to compute the basepoint sample indices $m(k)$ and the fractional sample advance $\mu(k)$.

        $$\mu(k + 1) = \mu(k) + \frac{1}{r}$$

        If $\mu(k + 1) > 1$, then the basepoint sample index $m(k + 1) = m(k) + 1$ and $\mu(k + 1) = \mu(k + 1) - 1$.
        Otherwise, the basepoint sample index $m(k + 1) = m(k)$.

    References:
        - Michael Rice, *Digital Communications: A Discrete Time Approach*, Section 8.4.2.
        - `Qasim Chaudhari, Fractional Delay Filters Using the Farrow Structure.
          <https://wirelesspi.com/fractional-delay-filters-using-the-farrow-structure/>`_

    Examples:
        .. ipython:: python

            d = np.zeros(10, dtype=float); \
            d[d.size // 2] = 1

            t = np.linspace(-2, 2, 1000); \
            sinc = np.sinc(t)

            plt.figure(); \
            sdr.plot.time_domain(t, sinc, color="k", linestyle="--", label="Ideal Sinc");
            for order in range(1, 5 + 1, 2):
                rate = 100
                farrow = sdr.FarrowResampler(order)
                h = farrow(d, rate, mode="full")
                sdr.plot.time_domain(h, sample_rate=rate, offset=-d.size // 2 - farrow.delay, label=order)
            @savefig sdr_FarrowResampler_1.svg
            plt.legend(title="Farrow Order"); \
            plt.xlim(-2, 2); \
            plt.xlabel("Samples, $n$"); \
            plt.title("Farrow Resampler Impulse Responses");

        Create a sine wave with angular frequency $\omega = 2 \pi / 5.179$. Interpolate the signal by
        $r = \pi$ using Farrow piecewise polynomial Farrow resamplers.

        .. ipython:: python

            x = np.cos(2 * np.pi / 5.179 * np.arange(11))
            rate = np.pi

        Compare the outputs of the Farrow resamplers with varying polynomial order. The convolution mode is set to
        `"rate"`, which means that the output signal $y[k]$ is aligned with the input signal.

        .. ipython:: python

            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input");
            for order in range(1, 5 + 1, 2):
                farrow = sdr.FarrowResampler(order)
                y = farrow(x, rate, mode="rate")
                sdr.plot.time_domain(y, sample_rate=rate, marker=".", label=f"Output {order}")
            @savefig sdr_FarrowResampler_2.svg
            plt.title("Farrow Resampler Outputs");

        Compare the outputs of the Farrow resamplers with varying polynomial order. The convolution mode is set to
        `"full"`, which means that the output signal $y[k]$ is delayed by the filter delay $D$. The plot is offset by
        the delay for easier viewing.

        .. ipython:: python

            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input");
            for order in range(1, 5 + 1, 2):
                farrow = sdr.FarrowResampler(order)
                y = farrow(x, rate, mode="full")
                sdr.plot.time_domain(y, sample_rate=rate, offset=-farrow.delay, marker=".", label=f"Output {order}")
            @savefig sdr_FarrowResampler_3.svg
            plt.title("Farrow Resampler Outputs");

        Run a Farrow resampler with cubic polynomial order in streaming mode.

        .. ipython:: python

            x = np.cos(2 * np.pi / 5.179 * np.arange(40))
            farrow = sdr.FarrowResampler(3, streaming=True)

            y = []
            for i in range(0, x.size, 10):
                yi = farrow(x[i : i + 10], rate, mode="rate")
                y.append(yi)
            y = np.concatenate(y)

            @savefig sdr_FarrowResampler_4.svg
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=rate, marker=".", label="Cubic concatenated"); \
            plt.title("Cubic Farrow Resampler Concatenated Outputs");

        Run a Farrow resampler with cubic polynomial order in streaming mode by clocking 10 outputs at a time.
        Clocking a specific number of output samples is useful in a demodulator when a specific number of samples per
        symbol are requested at a given resampling rate.

        .. ipython:: python

            farrow = sdr.FarrowResampler(3, streaming=True)

            n_outputs = 10; \
            i = 0; \
            y = []; \
            lengths = [];
            while i < x.size - n_outputs / rate:
                yi, n_inputs = farrow.clock_outputs(x[i:], rate, n_outputs, mode="rate")
                i += n_inputs
                y.append(yi)
                lengths.append(yi.size)
            y = np.concatenate(y); \
            print(lengths)

            @savefig sdr_FarrowResampler_5.svg
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=rate, marker=".", label="Cubic concatenated"); \
            plt.title("Cubic Farrow Resampler Concatenated Outputs");

        See the :ref:`farrow-arbitrary-resampler` example.

    Group:
        dsp-arbitrary-resampling
    """

    def __init__(self, order: int, alpha: float = 0.5, streaming: bool = False):
        r"""
        Creates a new Farrow arbitrary resampler.

        Arguments:
            order: The order $p$ of the Lagrange interpolating polynomial.
            alpha: A free design parameter $0 \le \alpha \le 1$ that controls the shape of a 2nd order filter.
                This ensures that the filter has an even number of taps and is linear phase. The default value
                is $\alpha = 0.5$, which is a good compromise between performance and fixed-point computational
                complexity. It was found through simulation that $\alpha = 0.43$ is optimal for BPSK using a
                square root raised cosine filter with 100% excess bandwidth.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~FarrowResampler.__call__()`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        super().__init__(order, alpha, streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike, rate: npt.ArrayLike, mode: Literal["rate", "full"] = "rate") -> npt.NDArray:
        r"""
        Resamples the input signal $x[n]$ by the given arbitrary rate $r$.

        $$x[n] = x(n T_s)$$
        $$y[n] = x(n T_s / r)$$

        Arguments:
            x: The input signal $x[n] = x(n T_s)$ with length $L$.
            rate: The resampling rate $r$. The rate can either be a scalar or an array of the same size as
                the input signal $x[n]$.
            mode: The convolution mode.

                - `"rate"`: The output signal $y[k]$ is aligned with the input signal, such that $y[n] = x[n / r]$.

                  In non-streaming mode, $(L - D) \cdot r$ output samples are produced. In streaming mode, the first
                  call returns $(L - D) \cdot r$ output samples. On subsequent calls, $L \cdot r$ output samples are
                  produced.

                - `"full"`: The full convolution is performed, and the filter delay $D$ is observed, such that
                  $y[n] = x[(n - D) / r]$.

                  In non-streaming mode, $L \cdot r$ output samples are produced. In streaming mode, each call returns
                  $L \cdot r$ output samples.

        Returns:
            The resampled signal $y[n]$.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        x = verify_arraylike(x, complex=True, atleast_1d=True, ndim=1)
        rate = verify_arraylike(rate, float=True, positive=True, atleast_1d=True, sizes=[1, x.size])
        verify_literal(mode, ["rate", "full"])

        x, rate = np.broadcast_arrays(x, rate)
        x.setflags(write=False)
        rate.setflags(write=False)

        # NOTE: This function will return 1 more m and mu value. That extra value is the next state.
        m_max = x.size
        m, mu = _process_mu_fixed_inputs(self._m_next, self._mu_next, rate, m_max)

        if self.streaming:
            # Save the next m and mu state
            self._m_next = m[-1] - m_max
            self._mu_next = mu[-1]

        # Remove the next m and mu from the end of the array
        m = m[:-1]
        mu = mu[:-1]

        # Pass the computed m's and mu's to the fractional delay Farrow
        y = super().__call__(x, m, mu, mode=mode)

        return convert_output(y)

    def clock_outputs(
        self, x: npt.ArrayLike, rate: npt.ArrayLike, n_outputs: int, mode: Literal["rate", "full"] = "rate"
    ) -> tuple[npt.NDArray, int]:
        r"""
        Resamples the input signal $x[n]$ by the given arbitrary rate $r$.

        $$x[n] = x(n T_s)$$
        $$y[n] = x(n T_s / r)$$

        Arguments:
            x: The input signal $x[n] = x(n T_s)$ with length $L$.
            rate: The resampling rate $r$. The rate can either be a scalar or an array of the same size as
                the input signal $x[n]$.
            n_outputs: The requested number of output samples in $y[n]$.
            mode: The convolution mode.

                - `"rate"`: The output signal $y[k]$ is aligned with the input signal, such that $y[n] = x[n / r]$.
                - `"full"`: The full convolution is performed, and the filter delay $D$ is observed, such that
                  $y[n] = x[(n - D) / r]$.

        Returns:
            - The resampled signal $y[n]$.
            - The number of processed input samples `n_inputs`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        x = verify_arraylike(x, complex=True, atleast_1d=True, ndim=1)
        rate = verify_arraylike(rate, float=True, positive=True, atleast_1d=True, sizes=[1, x.size])
        verify_scalar(n_outputs, int=True, positive=True)
        verify_literal(mode, ["rate", "full"])

        # We don't want FarrowFractionalDelay to manage m's and mu's for the next call. We want it to process all the
        # m's and mu's we pass to it. Because of this, we pass an extra "lookahead" samples for the "rate" mode.
        assert self._m_state.size == 0

        x, rate = np.broadcast_arrays(x, rate)
        x.setflags(write=False)
        rate.setflags(write=False)

        # NOTE: This function will return 1 more m and mu value. That extra value is the next state.
        m, mu = _process_mu_fixed_outputs(self._m_next, self._mu_next, rate, n_outputs)

        n_inputs = m[-2] + 1  # The number of inputs required
        if mode == "rate":
            # Pass extra samples on this call so that FarrowFractionalDelay can process all the m's and mu's we provide.
            n_inputs += self.delay

        if self.streaming:
            # Save the next m and mu state
            self._m_next = m[-1] - n_inputs
            self._mu_next = mu[-1]

        # Remove the next m and mu from the end of the array
        assert n_inputs <= x.size
        x = x[:n_inputs]
        m = m[:-1]
        mu = mu[:-1]

        # Pass the computed m's and mu's to the fractional delay Farrow
        y = super().__call__(x, m, mu, mode=mode)

        return convert_output(y), n_inputs

    ##############################################################################
    # Streaming mode
    ##############################################################################

    def reset(self):
        super().reset()

        # Initial fractional sample delay accounts for filter delay
        self._m_next = 0
        self._mu_next = 0

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def delay(self) -> int:
        r"""
        The delay $D$ of the Farrow FIR filters in samples.

        The delay is only observed when the convolution mode is set to `"full"`. No delay is observed when the
        convolution mode is set to `"rate"`. Due the multirate nature of the Farrow resampler, output sample
        $D \cdot r$ corresponds to the first input sample, where $r$ is the current resampling rate.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return super().delay


def _compute_lagrange_basis(order: int) -> tuple[int, npt.NDArray, npt.NDArray]:
    """
    Computes the Lagrange basis polynomials for the Farrow filter.
    """
    # Support points (e.g., for 4-tap Lagrange interpolation centered at x = 0)
    # Indices for x[n-1], x[n], x[n+1], x[n+2]
    lookahead = order // 2
    x_points = np.arange(order + 1) - lookahead

    # Store basis polynomials
    basis_polys = np.zeros((order + 1, order + 1), dtype=float)

    # Compute all Lagrange basis polynomials l_k(mu)
    for k in range(x_points.size):
        y_basis = np.zeros_like(x_points)
        y_basis[k] = 1  # delta function at position k
        poly_k = scipy.interpolate.lagrange(x_points, y_basis)
        basis_polys[k, :] = poly_k.coeffs  # Degree-descending order

    # Compute Farrow coefficients. Convert
    farrow_coeffs = basis_polys.transpose()

    return lookahead, basis_polys, farrow_coeffs


@numba.jit(nopython=True, cache=True)
def _process_mu_fixed_inputs(
    m_next: int, mu_next: float, rate: npt.NDArray, m_max: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Determines a series of basepoint sample indices `m` and fractional sample indices `mu` to achieve a sample rate
    increase of `rate`.
    """
    # Determine how many outputs are required, given the number of inputs requested
    n_outputs = int(np.ceil((m_max - m_next) * np.max(rate)))

    m = np.zeros(n_outputs + 1, dtype=np.int64)  # Basepoint sample indices
    m[0] = m_next

    mu = np.zeros(n_outputs + 1, dtype=np.float64)  # Fractional sample indices
    mu[0] = mu_next

    for i in range(0, n_outputs):
        if m[i] >= m_max:
            # m[i] is current m_next
            m = m[: i + 1]
            mu = mu[: i + 1]
            break

        # Accumulate the fractional sample index by 1 / rate
        mu[i + 1] = mu[i] + 1 / rate[m[i]]

        # Set the basepoint sample index the same
        m[i + 1] = m[i]

        # Handle overflows in the fractional part
        if mu[i + 1] >= 1:
            overflow = int(mu[i + 1])
            mu[i + 1] -= overflow  # Reset mu back to [0, 1)
            m[i + 1] += overflow  # Move the basepoint to the next sample

    return m, mu


@numba.jit(nopython=True, cache=True)
def _process_mu_fixed_outputs(
    m_next: int, mu_next: float, rate: npt.NDArray, n_outputs: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Determines a series of basepoint sample indices `m` and fractional sample indices `mu` to achieve a sample rate
    increase of `rate`.
    """
    m = np.zeros(n_outputs + 1, dtype=np.int64)  # Basepoint sample indices
    m[0] = m_next

    mu = np.zeros(n_outputs + 1, dtype=np.float64)  # Fractional sample indices
    mu[0] = mu_next

    for i in range(0, n_outputs):
        # Accumulate the fractional sample index by 1 / rate
        mu[i + 1] = mu[i] + 1 / rate[m[i]]

        # Set the basepoint sample index the same
        m[i + 1] = m[i]

        # Handle overflows in the fractional part
        if mu[i + 1] >= 1:
            overflow = int(mu[i + 1])
            mu[i + 1] -= overflow  # Reset mu back to [0, 1)
            m[i + 1] += overflow  # Move the basepoint to the next sample

    return m, mu
