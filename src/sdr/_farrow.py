"""
A module containing a Farrow arbitrary resampler.
"""

from __future__ import annotations

from typing import Any, overload

import numba
import numpy as np
import numpy.typing as npt
import scipy.interpolate
import scipy.signal
from typing_extensions import Literal

from ._helper import convert_output, export, verify_arraylike, verify_bool, verify_literal, verify_scalar


def _compute_lagrange_basis(order: int) -> tuple[int, npt.NDArray, npt.NDArray]:
    """
    Computes the Lagrange basis polynomials for the Farrow filter.
    """
    # Support points (e.g., for 4-tap Lagrange interpolation centered at x = 0)
    # Indices for x[n-1], x[n], x[n+1], x[n+2]
    delay = order // 2
    x_points = np.arange(order + 1) - delay

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

    return delay, basis_polys, farrow_coeffs


@export
class FarrowFractionalDelay:
    r"""
    Implements a piecewise polynomial Farrow fractional delay filter.

    Consider the series of input samples $x[n-D], \ldots, x[n], \ldots, x[n+p-D]$. The goal is to interpolate the
    value of $x(t)$ at the time $t = n + \mu$. The Lagrange polynomial interpolation is computed as

    $$x(n + \mu) = \sum_{k=0}^{p} x[n + k - D] \cdot \ell_k(\mu) ,$$

    where $D$ is the delay offset (usually so the center tap is at zero) and $\ell_k(\mu)$ is the Lagrange basis
    polynomial corresponding to tap $k$. Note that $\ell_k(\mu)$ is independent of the data and only depends on the
    fractional shift $\mu$.

    The $k$-th Lagrange basis polynomial is defined as

    $$\ell_k(\mu) = \prod_{\substack{j=0 \\ j \ne k}}^{p} \frac{\mu - j}{k - j} ,$$

    which is a polynomial of degree $p$ in $\mu$. The Lagrange basis polynomials are 1 at $\mu = k$ and 0 at
    $\mu = j$ for $j \ne k$.

    It is often desirable to change $\mu$ on the fly, which means that the Lagrange basis polynomials must be
    recomputed for each new value of $\mu$. Instead, the Farrow structure selects the coefficients of each
    Lagrange basis polynomial for a given degree of $\mu$ and forms it into a FIR filter

    $$h_m[n] = \{\ell_{0,m}, \ell_{1,m}, \ldots, \ell_{p,m}\} ,$$

    where $h_m[n]$ is the $m$-th FIR filter and $\ell_{k,m}$ is the coefficient of $\mu^m$ in the $k$-th Lagrange
    basis polynomial.

    The output of the $m$-th FIR filter is $z_m[n] = h_m[n] * x[n]$. The Farrow structure allows the interpolation
    calculation to be reorganized as

    $$x(n + \mu) = \sum_{k=0}^{p} x[n + k - D] \cdot \ell_k(\mu) = \sum_{k=0}^{p} z_m[n] \cdot \mu^k .$$

    which can be efficiently computed using Horner's method.

    $$x(n + \mu) = \sum_{k=0}^{p} z_m[k] \cdot \mu^k = z_0[n] + \mu \cdot (z_1[n] + \mu \cdot (z_2[n] + \ldots)).$$

    .. note::

        In Michael Rice's book, the Farrow filter is described as a fractional advance. In this implementation, we
        consider the Farrow filter as a fractional delay.

        $$y_{\text{book}}[k] = x((m_{\text{book}}(k) + \mu_{\text{book}}(k)) T_s) = x[m_{\text{book}}(k) + \mu_{\text{book}}(k)]$$
        $$m_{\text{book}}(k) = m(k) - 1$$
        $$\mu_{\text{book}}(k) = 1 - \mu(k)$$
        $$y[k] = x((m(k) - \mu(k)) T_s) = x[m(k) - \mu(k)]$$

    References:
        - Michael Rice, *Digital Communications: A Discrete Time Approach*, Section 8.4.2.
        - `Qasim Chaudhari, Fractional Delay Filters Using the Farrow Structure.
          <https://wirelesspi.com/fractional-delay-filters-using-the-farrow-structure/>`_

    Examples:
        Plot the 3rd order Lagrange basis polynomials. Notice that each polynomial is 1 and 0 at the corresponding
        indices.

        .. ipython:: python

            farrow = sdr.FarrowFractionalDelay(3)
            mu = np.linspace(0, farrow.order, 1000) - farrow.delay

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
            sdr.plot.time_domain(x, offset=-farrow.delay, marker=".", linestyle="none", label="Input"); \
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
            plt.title(f"Fractional delay {mu} samples");

            @savefig sdr_FarrowFractionalDelay_4.svg
            mu = 0.5; \
            plt.figure(); \
            sdr.plot.time_domain(x, marker=".", color="k"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(1)(x, mu=mu), label="Farrow 1"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(2)(x, mu=mu), label="Farrow 2"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(3)(x, mu=mu), label="Farrow 3"); \
            plt.title(f"Fractional delay {mu} samples");

            @savefig sdr_FarrowFractionalDelay_5.svg
            mu = 1; \
            plt.figure(); \
            sdr.plot.time_domain(x, marker=".", color="k"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(1)(x, mu=mu), label="Farrow 1"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(2)(x, mu=mu), label="Farrow 2"); \
            sdr.plot.time_domain(sdr.FarrowFractionalDelay(3)(x, mu=mu), label="Farrow 3"); \
            plt.title(f"Fractional delay {mu} samples");

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
                preserved between calls to :meth:`~FarrowResampler.__call__()`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        self._order = verify_scalar(order, int=True, positive=True)
        self._alpha = verify_scalar(alpha, float=True, inclusive_min=0, inclusive_max=1)
        self._streaming = verify_bool(streaming)
        self._state: npt.NDArray  # FIR filter state. Will be updated in reset().

        self._delay, self._lagrange_polys, self._taps = _compute_lagrange_basis(self._order)
        self._lookahead = self._taps.shape[1] - self._delay - 1

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
        Applies the fractional sample delay $\mu(k)$ to the input signal $x[n]$ at the given basepoint sample
        indices $m(k)$.

        $$y[k] = x((m(k) - \mu(k)) T_s) = x[m(k) - \mu(k)]$$

        Arguments:
            x: The input signal $x[n] = x(n T_s)$.
            m: The basepoint sample indices $m(k)$, which are the integer sample indices of the input signal.
            mu: The fractional sample indices $0 \le \mu(k) \le 1$, which is the fractional sample delay of the
                input signal at input sample $m(k)$.
            mode: The non-streaming convolution mode.

                - `"rate"`: The output signal $y[k]$ has length $L \cdot r$, proportional to the resampling rate
                  $r$. Output sample 0 aligns with input sample 0.
                - `"full"`: The full convolution is performed. The output signal $y[k]$ has length $(L + N) \cdot r$,
                  where $N$ is the order of the prototype filter. Output sample :obj:`~FarrowFractionalDelay.delay`
                  aligns with input sample 0.

                In streaming mode, the `"full"` convolution is performed. However, for each $L$ input samples
                only $L \cdot r$ output samples are produced per call. A final call to :meth:`~FarrowFractionalDelay.flush()`
                is required to flush the filter state.

        Returns:
            The resampled signal $y[k]$.

        Notes:
            If using streaming mode, the filter output is delayed by the filter delay $d$, see
            :obj:`~FarrowFractionalDelay.delay`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        x = verify_arraylike(x, complex=True, atleast_1d=True, ndim=1)
        if m is None:
            m = np.arange(0, x.size)
        else:
            m = verify_arraylike(m, int=True, inclusive_min=0, exclusive_max=x.size, atleast_1d=True, ndim=1)
        if mu is None:
            mu = np.zeros_like(x, dtype=float)
        else:
            mu = verify_arraylike(mu, float=True, inclusive_min=0, inclusive_max=1, atleast_1d=True, ndim=1)
        verify_literal(mode, ["rate", "full"])

        # m and mu can be scalars or arrays. If they are arrays, they must be the same size. This NumPy function
        # should error if the sizes are not broadcast-able.
        m, mu = np.broadcast_arrays(m, mu)

        # Prepend the pervious state (zeros initially)
        x = np.concatenate((self._state, x))
        m = np.concatenate((self._m_state, m))
        mu = np.concatenate((self._mu_state, mu))

        # Compute the FIR filter outputs for the entire input signal
        z = []
        for i in range(self.order + 1):
            zi = scipy.signal.convolve(x, self._taps[i, :], mode="full")
            z.append(zi)

        # Add the filter delay due to historical inputs
        # NOTE: If non-streaming mode, state is always empty
        m += self._state.size

        if mode == "rate":
            # Account for the Farrow filter delay so that the first output sample is aligned with the first input
            # sample
            m += self._delay

            if self.streaming:
                # If in streaming mode, we will repeatedly call this function. A valid output correlates with some
                # previous inputs and some future inputs. Once we added the delay, some of the m values will be
                # outside the valid correlation zone. They correspond to correlation outputs with some zeros inputs
                # (from the "full" convolution). We need to only consider the valid m values during this call and
                # save the rest for the next call.

                # Save the m values needed to be processed next call. We also subtract from the m values so that
                # the state is in [-delay, 0) range.
                self._m_state = m[m >= x.size] - x.size - self._delay
                self._mu_state = mu[m >= x.size]

                # Keep only the valid m values for this call
                mu = mu[m < x.size]  # NOTE: Need to resize m last since it is used in the indexing
                m = m[m < x.size]

        # Interpolate the output samples using the Horner method
        y = z[0][m]
        for i in range(1, self.order + 1):
            y *= mu
            y += z[i][m]

        if self.streaming:
            # Save the last several inputs so we can use them in the next call
            self._state = x[-(self.taps.shape[1] - 1) :]

        return convert_output(y)

    def __repr__(self) -> str:
        return f"sdr.{type(self).__name__}({self.order}, streaming={self.streaming})"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  taps: {self.taps.shape} shape"
        string += f"\n    {np.array2string(self.taps, prefix='    ')}"
        string += f"\n  delay: {self.delay}"
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
        self._state = np.zeros(0, dtype=float)
        self._m_state = np.zeros(0, dtype=int)
        self._mu_state = np.zeros(0, dtype=float)

        self._first_call = True

    def flush(self) -> npt.NDArray:
        """
        Flushes the filter state by passing zeros through the filter. Only useful when using streaming mode.

        Returns:
            The remaining delayed signal $y[k]$.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.

        Group:
            Streaming mode only
        """
        x = np.zeros(self.delay, dtype=float)
        y = self(x)
        return y

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

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._delay


@export
class FarrowResampler(FarrowFractionalDelay):
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

            @savefig sdr_FarrowResampler_1.svg
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y1, sample_rate=rate, marker=".", label="Output"); \
            plt.title("Linear Farrow Resampler");

        Create a quadratic Farrow piecewise polynomial interpolator.

        .. ipython:: python

            farrow2 = sdr.FarrowResampler(2)
            y2 = farrow2(x, rate)

            @savefig sdr_FarrowResampler_2.svg
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y2, sample_rate=rate, marker=".", label="Output"); \
            plt.title("Quadratic Farrow Resampler");

        Create a cubic Farrow piecewise polynomial interpolator.

        .. ipython:: python

            farrow3 = sdr.FarrowResampler(3)
            y3 = farrow3(x, rate)

            @savefig sdr_FarrowResampler_3.svg
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y3, sample_rate=rate, marker=".", label="Output"); \
            plt.title("Cubic Farrow Resampler");

        Create a quartic Farrow piecewise polynomial interpolator.

        .. ipython:: python

            farrow4 = sdr.FarrowResampler(4)
            y4 = farrow4(x, rate)

            @savefig sdr_FarrowResampler_4.svg
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y4, sample_rate=rate, marker=".", label="Output"); \
            plt.title("Quartic Farrow Resampler");

        Compare the outputs of the Farrow resamplers with varying polynomial order.

        .. ipython:: python

            @savefig sdr_FarrowResampler_5.svg
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

            @savefig sdr_FarrowResampler_6.svg
            plt.figure(); \
            sdr.plot.time_domain(x, sample_rate=1, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=rate, offset=-farrow4.delay, marker=".", label="Quartic concatenated"); \
            plt.title("Quartic Farrow Resampler Concatenated Outputs");

        See the :ref:`farrow-arbitrary-resampler` example.

    Group:
        dsp-arbitrary-resampling
    """

    def __init__(self, order: int, alpha: float = 0.5, align: bool = True, streaming: bool = False):
        r"""
        Creates a new Farrow arbitrary resampler.

        Arguments:
            order: The order $p$ of the Lagrange interpolating polynomial.
            alpha: A free design parameter $0 \le \alpha \le 1$ that controls the shape of a 2nd order filter.
                This ensures that the filter has an even number of taps and is linear phase. The default value
                is $\alpha = 0.5$, which is a good compromise between performance and fixed-point computational
                complexity. It was found through simulation that $\alpha = 0.43$ is optimal for BPSK using a
                square root raised cosine filter with 100% excess bandwidth.
            align: Indicates whether to remove the filter delay. If `True`, the output signal is aligned with the
                input signal. If `False`, the output signal is not aligned with the input signal.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~FarrowResampler.__call__()`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        super().__init__(order, alpha, align, streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    @overload
    def __call__(
        self,
        x: npt.NDArray,  # TODO: Change to npt.ArrayLike once Sphinx has better overload support
        rate: float,
        n_outputs: int,
    ) -> tuple[npt.NDArray, int]: ...

    @overload
    def __call__(
        self,
        x: npt.NDArray,  # TODO: Change to npt.ArrayLike once Sphinx has better overload support
        rate: float,
        n_outputs: None = None,
    ) -> npt.NDArray: ...

    def __call__(self, x: Any, rate: Any, n_outputs: Any = None) -> Any:
        r"""
        Resamples the input signal $x[n]$ by the given arbitrary rate $r$.

        $$x[n] = x(n T_s)$$
        $$y[n] = x(n T_s / r)$$

        Arguments:
            x: The input signal $x[n] = x(n T_s)$.
            rate: The resampling rate $r$.
            n_outputs: The requested number of computed samples in $y[n]$. If specified, the number of processed
                samples of $x[n]$ is returned.

        Returns:
            - The resampled signal $y[n] = x(n T_s / r)$.
            - The number of processed input samples `n_inputs`. This is only returned if `n_outputs` is provided.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        x = verify_arraylike(x, complex=True, atleast_1d=True, ndim=1)
        verify_scalar(rate, float=True, positive=True)

        if n_outputs is None:
            # NOTE: This function will return 1 more m and mu value. That extra value is the next state.
            n_inputs = x.size
            m, mu = process_mu_fixed_inputs(self._m_next, self._mu_next, rate, n_inputs)

        else:
            verify_scalar(n_outputs, int=True, positive=True)

            # NOTE: This function will return 1 more m and mu value. That extra value is the next state.
            m, mu = process_mu_fixed_outputs(self._m_next, self._mu_next, rate, n_outputs)
            n_inputs = m[-2] + 1  # The number of inputs required
            x = x[:n_inputs]  # Reduce the input samples, so we only pass what's needed to the FarrowFractionalDelay

        if self.streaming:
            # Save the next m and mu state
            self._m_next = m[-1] - n_inputs
            self._mu_next = mu[-1]

        # Remove the next m and mu from the end of the array
        m = m[:-1]
        mu = mu[:-1]

        # Pass the computed m's and mu's to the fractional delay Farrow
        y = super().__call__(x, m, 1 - mu)

        if n_outputs is None:
            return convert_output(y)
        else:
            return convert_output(y), n_inputs

    ##############################################################################
    # Streaming mode
    ##############################################################################

    def reset(self, state: npt.ArrayLike | None = None):
        super().reset(state)

        # Initial fractional sample delay accounts for filter delay
        self._m_next = 0
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

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def delay(self) -> int:
        r"""
        The delay $D$ of the Farrow FIR filters in samples.

        Output sample $D \cdot r$, corresponds to the first input sample, where $r$ is the current resampling rate.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._delay


@numba.jit
def process_mu_fixed_inputs(m_next: int, mu_next: float, rate: float, n_inputs: int) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Determines a series of basepoint sample indices `m` and fractional sample indices `mu` to achieve a sample rate
    increase of `rate`.
    """
    # Determine how many outputs are required, given the number of inputs requested
    n_outputs = int(np.ceil((n_inputs + 1) * rate))

    m = np.zeros(n_outputs + 1, dtype=np.int64)  # Basepoint sample indices
    m[0] = m_next

    mu = np.zeros(n_outputs + 1, dtype=np.float64)  # Fractional sample indices
    mu[0] = mu_next

    for i in range(1, n_outputs + 1):
        # Accumulate the fractional sample index by 1 / rate
        mu[i] = mu[i - 1] + 1 / rate

        # Set the basepoint sample index the same
        m[i] = m[i - 1]

        # Handle overflows in the fractional part
        if mu[i] >= 1:
            overflow = int(mu[i])
            mu[i] -= overflow  # Reset mu back to [0, 1)
            m[i] += overflow  # Move the basepoint to the next sample

        if m[i] >= n_inputs:
            # m[i] is current m_next
            break

    m = m[: i + 1]
    mu = mu[: i + 1]

    return m, mu


@numba.jit
def process_mu_fixed_outputs(
    m_next: int, mu_next: float, rate: float, n_outputs: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Determines a series of basepoint sample indices `m` and fractional sample indices `mu` to achieve a sample rate
    increase of `rate`.
    """
    m = np.zeros(n_outputs + 1, dtype=np.int64)  # Basepoint sample indices
    m[0] = m_next

    mu = np.zeros(n_outputs + 1, dtype=np.float64)  # Fractional sample indices
    mu[0] = mu_next

    for i in range(1, n_outputs + 1):
        # Accumulate the fractional sample index by 1 / rate
        mu[i] = mu[i - 1] + 1 / rate

        # Set the basepoint sample index the same
        m[i] = m[i - 1]

        # Handle overflows in the fractional part
        if mu[i] >= 1:
            overflow = int(mu[i])
            mu[i] -= overflow  # Reset mu back to [0, 1)
            m[i] += overflow  # Move the basepoint to the next sample

    return m, mu
