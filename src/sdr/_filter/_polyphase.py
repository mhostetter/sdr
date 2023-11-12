"""
A module for interpolating finite impulse response (FIR) filters.
"""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._helper import export
from ._fir import FIR


@export
def multirate_taps(
    up: int,
    down: int = 1,
    half_length: int = 12,
    A_stop: float = 80,
) -> npt.NDArray[np.float_]:
    r"""
    Computes the multirate FIR filter that achieves rational resampling by $P/Q$.

    Note:
        This filter can be used with :class:`sdr.Interpolator`, :class:`sdr.Decimator`, or :class:`sdr.Resampler`.

    Arguments:
        up: The interpolation rate $P$.
        down: The decimation rate $Q$.
        half_length: The half-length of the polyphase filters.
        A_stop: The stopband attenuation $A_{\text{stop}}$ in dB.

    Returns:
        The multirate FIR filter $h[n]$.

    References:
        - fred harris, *Multirate Signal Processing for Communication Systems*, Chapter 7: Resampling Filters
        - https://www.mathworks.com/help/dsp/ref/designmultiratefir.html

    Examples:
        Design a multirate FIR filter for rational resampling by 11/3.

        .. ipython:: python

            h = sdr.multirate_taps(11, 3)

            @savefig sdr_multirate_fir_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h);

            @savefig sdr_multirate_fir_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.magnitude_response(h);

    Group:
        dsp-multirate-filtering
    """
    if not isinstance(up, int):
        raise TypeError(f"Argument 'up' must be an integer, not {up}.")
    if not up >= 1:
        raise ValueError(f"Argument 'up' must be at least 1, not {up}.")
    P = up

    if not isinstance(down, int):
        raise TypeError(f"Argument 'down' must be an integer, not {down}.")
    if not down >= 1:
        raise ValueError(f"Argument 'down' must be at least 1, not {down}.")
    Q = down

    B = P if P > 1 else Q  # The number of polyphase branches
    R = max(P, Q)  # Inverse of the filter's fractional bandwidth

    # Compute the filter order, which is length - 1
    N = 2 * half_length * B

    # Compute ideal lowpass filter
    n = np.arange(N + 1)
    h = P / R * np.sinc((n - N // 2) / R)

    # Compute Kaiser window
    # beta = scipy.signal.windows.kaiser_beta(A_stop)
    if A_stop >= 50:
        beta = 0.1102 * (A_stop - 8.71)  # TODO: MATLAB uses 8.71 and SciPy uses 8.7
    elif A_stop > 21:
        beta = 0.5842 * (A_stop - 21) ** 0.4 + 0.07886 * (A_stop - 21)
    else:
        beta = 0
    w = scipy.signal.windows.kaiser(N + 1, beta)

    # Compute windowed filter
    h = h * w

    if not (Q > P > 1 and (half_length * P) % Q != 0):
        # The first and last elements are zero. Remove the last zero so that the filter is evenly
        # partitioned across the polyphase branches. The filter now has length 2 * half_length * B and
        # each polyphase branch has length 2 * half_length.
        h = h[:-1]

    # If the above condition is not true, the first and last elements are non-zero. The filter length
    # is 2 * half_length * B + 1 and each polyphase branch has length 2 * half_length + 1. The final
    # column in the polyphase matrix will be padded with zeros.

    return h


def _multirate_taps_linear(rate: int) -> npt.NDArray[np.float_]:
    r"""
    The multirate filter is designed to linearly interpolate between samples. The filter coefficients are a
    length-$2r$ linear ramp $\frac{1}{r} [0, ..., r-1, r, r-1, ..., 1]$. The first output sample aligns with the
    first input sample.
    """
    h = np.zeros(2 * rate, dtype=float)
    h[:rate] = np.arange(0, rate) / rate
    h[rate:] = np.arange(rate, 0, -1) / rate
    return h


def _multirate_taps_linear_matlab(rate: int) -> npt.NDArray[np.float_]:
    r"""
    The multirate filter is designed to linearly interpolate between samples. The filter coefficients are a
    length-$2r$ linear ramp $\frac{1}{r} [1, ..., r-1, r, r-1, ..., 0]$. This is method MATLAB uses. The first
    output sample is advanced from the first input sample.
    """
    h = np.zeros(2 * rate, dtype=float)
    h[:rate] = np.arange(1, rate + 1) / rate
    h[rate:] = np.arange(rate - 1, -1, -1) / rate
    return h


def _multirate_taps_zoh(rate: int) -> npt.NDArray[np.float_]:
    """
    The multirate filter is designed to be a zero-order hold. The filter coefficients are a length-$r$ array of ones.
    """
    h = np.ones(rate, dtype=float)
    return h


@export
def polyphase_matrix(up: int, down: int, taps: npt.ArrayLike) -> npt.NDArray:
    """
    Converts the multirate FIR filter taps $h_i$ into the polyphase matrix $H_{i, j}$ that achieves
    rational resampling by $P/Q$.

    Arguments:
        up: The interpolation rate $P$.
        down: The decimation rate $Q$.
        taps: The multirate FIR filter taps $h_i$.

    Returns:
        The polyphase matrix $H_{i, j}$.

    Notes:
        The multirate FIR filter taps $h_i$ are arranged down the columns of the polyphase matrix
        $H_{i, j}$ as follows:

        .. code-block:: text
            :caption: Polyphase Matrix

            +------+------+------+------+
            | h[0] | h[3] | h[6] | h[9] |
            +------+------+------+------+
            | h[1] | h[4] | h[7] | 0    |
            +------+------+------+------+
            | h[2] | h[5] | h[8] | 0    |
            +------+------+------+------+

    References:
        - fred harris, *Multirate Signal Processing for Communication Systems*, Chapter 7: Resampling Filters

    Examples:
        Convert the multirate FIR filter (notional taps for demonstration) into a polyphase matrix
        for rational resampling by 3/4 and 1/6.

        .. ipython:: python

            h = np.arange(0, 20)
            sdr.polyphase_matrix(3, 4, h)
            sdr.polyphase_matrix(1, 6, h)

    Group:
        dsp-multirate-filtering
    """
    if not isinstance(up, int):
        raise TypeError(f"Argument 'up' must be an integer, not {up}.")
    if not up >= 1:
        raise ValueError(f"Argument 'up' must be at least 1, not {up}.")
    P = up

    if not isinstance(down, int):
        raise TypeError(f"Argument 'down' must be an integer, not {down}.")
    if not down >= 1:
        raise ValueError(f"Argument 'down' must be at least 1, not {down}.")
    Q = down

    taps = np.asarray(taps)

    B = P if P > 1 else Q  # The number of polyphase branches
    N = math.ceil(taps.size / B) * B  # Filter length

    # Append zeros to the end of the taps so that the filter length is a multiple of B
    taps_pad = np.pad(taps, (0, N - taps.size), mode="constant")

    # Reshape the taps down the columns of H
    H = taps_pad.reshape(-1, B).T

    return H


def _polyphase_interpolate(
    x: npt.NDArray,
    state: npt.NDArray,
    H: npt.NDArray,
    mode: Literal["rate", "full"],
    streaming: bool,
) -> tuple[npt.NDArray, npt.NDArray]:
    B, M = H.shape  # Number of polyphase branches, polyphase filter length
    dtype = np.result_type(x, H)

    if streaming:
        x_pad = np.concatenate((state, x))  # Prepend previous inputs from last __call__() call
        Y = np.zeros((B, x.size), dtype=dtype)
        for i in range(B):
            Y[i] = scipy.signal.convolve(x_pad, H[i], mode="valid")
        state = x_pad[-(M - 1) :]
    else:
        if mode == "full":
            Y = np.zeros((B, x.size + M - 1), dtype=dtype)
            for i in range(B):
                Y[i] = scipy.signal.convolve(x, H[i], mode="full")
        elif mode == "rate":
            Y = np.zeros((B, x.size), dtype=dtype)
            for i in range(B):
                corr = scipy.signal.convolve(x, H[i], mode="full")
                Y[i] = corr[M // 2 : M // 2 + x.size]
        else:
            raise ValueError(f"Argument 'mode' must be 'rate' or 'full', not {mode!r}.")

    # Commutate the outputs of the polyphase filters
    y = Y.T.flatten()

    return y, state


@export
class Interpolator(FIR):
    r"""
    Implements a polyphase interpolating FIR filter.

    Notes:
        The polyphase interpolating filter is equivalent to first upsampling the input signal $x[n]$ by $r$
        (by inserting $r-1$ zeros between each sample) and then filtering the upsampled signal with the
        prototype FIR filter with feedforward coefficients $h_{i}$.

        Instead, the polyphase interpolating filter first decomposes the prototype FIR filter into $r$ polyphase
        filters with feedforward coefficients $h_{i, j}$. The polyphase filters are then applied to the
        input signal $x[n]$ in parallel. The output of the polyphase filters are then commutated to produce
        the output signal $y[n]$. This prevents the need to multiply with zeros in the upsampled input,
        as is needed in the first case.

        .. code-block:: text
           :caption: Polyphase 2x Interpolating FIR Filter Block Diagram

                                  +------------------------+
                              +-->| h[0], h[2], h[4], h[6] |--> ..., y[2], y[0]
                              |   +------------------------+
            ..., x[1], x[0] --+
                              |   +------------------------+
                              +-->| h[1], h[3], h[5], 0    |--> ..., y[3], y[1]
                                  +------------------------+

            Input Hold                                          Output Commutator
                                                                (top-to-bottom)

            x[n] = Input signal with sample rate fs
            y[n] = Output signal with sample rate fs * r
            h[n] = Prototype FIR filter

        The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

        $$h_{i, j} = h_{i + j r} .$$

    Examples:
        Create an input signal to interpolate.

        .. ipython:: python

            x = np.cos(np.pi / 4 * np.arange(40))

        Create a polyphase filter that interpolates by 7 using the Kaiser window method.

        .. ipython:: python

            fir = sdr.Interpolator(7); fir
            y = fir(x)

            @savefig sdr_Interpolator_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, marker=".", label="Interpolated"); \
            plt.title("Interpolation by 7 with the Kaiser window method"); \
            plt.tight_layout();

        Create a streaming polyphase filter that interpolates by 7 using the Kaiser window method. This filter
        preserves state between calls.

        .. ipython:: python

            fir = sdr.Interpolator(7, streaming=True); fir

            y1 = fir(x[0:10]); \
            y2 = fir(x[10:20]); \
            y3 = fir(x[20:30]); \
            y4 = fir(x[30:40]); \
            y5 = fir.flush()

            @savefig sdr_Interpolator_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y1, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 0, marker=".", label="Interpolated $y_1[n]$"); \
            sdr.plot.time_domain(y2, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 10, marker=".", label="Interpolated $y_2[n]$"); \
            sdr.plot.time_domain(y3, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 20, marker=".", label="Interpolated $y_3[n]$"); \
            sdr.plot.time_domain(y4, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 30, marker=".", label="Interpolated $y_4[n]$"); \
            sdr.plot.time_domain(y5, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 40, marker=".", label="Interpolated $y_5[n]$"); \
            plt.title("Streaming interpolation by 7 with the Kaiser window method"); \
            plt.tight_layout();

        Create a polyphase filter that interpolates by 7 using linear method.

        .. ipython:: python

            fir = sdr.Interpolator(7, "linear"); fir
            y = fir(x)

            @savefig sdr_Interpolator_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, marker=".", label="Interpolated"); \
            plt.title("Interpolation by 7 with the linear method"); \
            plt.tight_layout();

        Create a polyphase filter that interpolates by 7 using the zero-order hold method. It is recommended to use
        the `"full"` convolution mode. This way the first upsampled symbol has $r$ samples.

        .. ipython:: python

            fir = sdr.Interpolator(7, "zoh"); fir
            y = fir(x, mode="full")

            @savefig sdr_Interpolator_4.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, offset=-fir.delay/fir.rate, marker=".", label="Interpolated"); \
            plt.title("Interpolation by 7 with the zero-order hold method"); \
            plt.tight_layout();

    Group:
        dsp-multirate-filtering
    """

    def __init__(
        self,
        rate: int,
        taps: Literal["kaiser", "linear", "linear-matlab", "zoh"] | npt.ArrayLike = "kaiser",
        streaming: bool = False,
    ):
        r"""
        Creates a polyphase FIR interpolating filter.

        Arguments:
            rate: The interpolation rate $r$.
            taps: The multirate filter design specification.

                - `"kaiser"`: The multirate filter is designed using :func:`~sdr.multirate_taps()`
                  with arguments `rate` and 1.
                - `"linear"`: The multirate filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2r$ linear ramp $\frac{1}{r} [0, ..., r-1, r, r-1, ..., 1]$.
                  The first output sample aligns with the first input sample.
                - `"linear-matlab"`: The multirate filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2r$ linear ramp $\frac{1}{r} [1, ..., r-1, r, r-1, ..., 0]$.
                  This is method MATLAB uses. The first output sample is advanced from the first input sample.
                - `"zoh"`: The multirate filter is designed to be a zero-order hold.
                  The filter coefficients are a length-$r$ array of ones.
                - `npt.ArrayLike`: The multirate filter feedforward coefficients $h_i$.

            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~Interpolator.__call__()`.
        """
        if not isinstance(rate, int):
            raise TypeError("Argument 'rate' must be an integer.")
        if not rate >= 1:
            raise ValueError(f"Argument 'rate' must be at least 1, not {rate}.")
        self._rate = rate
        self._method: Literal["kaiser", "linear", "linear-matlab", "zoh", "custom"]

        if not isinstance(taps, str):
            self._method = "custom"
            taps = np.asarray(taps)
        elif taps == "kaiser":
            self._method = "kaiser"
            taps = multirate_taps(rate, 1)
        elif taps == "linear":
            self._method = "linear"
            taps = _multirate_taps_linear(rate)
        elif taps == "linear-matlab":
            self._method = "linear-matlab"
            taps = _multirate_taps_linear_matlab(rate)
        elif taps == "zoh":
            self._method = "zoh"
            taps = _multirate_taps_zoh(rate)
        else:
            raise ValueError(
                f"Argument 'taps' must be 'kaiser', 'linear', 'linear-matlab', 'zoh', or an array-like, not {taps}."
            )

        self._polyphase_taps = polyphase_matrix(rate, 1, taps)

        super().__init__(taps, streaming=streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike, mode: Literal["rate", "full"] = "rate") -> npt.NDArray:
        r"""
        Interpolates and filters the input signal $x[n]$ with the polyphase FIR filter.

        Arguments:
            x: The input signal $x[n]$ with sample rate $f_s$ and length $L$.
            mode: The non-streaming convolution mode.

                - `"rate"`: The output signal $y[n]$ has length $L r$ proportional to the interpolation rate $r$.
                  Output sample 0 aligns with input sample 0.
                - `"full"`: The full convolution is performed. The output signal $y[n]$ has length $(L + N) r$,
                  where $N$ is the order of the multirate filter. Output sample :obj:`~sdr.Interpolator.delay` aligns
                  with input sample 0.

                In streaming mode, the `"full"` convolution is performed. However, for each $L$ input samples
                only $L r$ output samples are produced per call. A final call to :meth:`~Interpolator.flush()`
                is required to flush the filter state.

        Returns:
            The filtered signal $y[n]$ with sample rate $f_s r$. The output length is dictated by
            the `mode` argument.

        Examples:
            Create an input signal to interpolate.

            .. ipython:: python

                x = np.cos(np.pi / 4 * np.arange(20))

            Interpolate the signal using the `"same"` mode.

            .. ipython:: python

                fir = sdr.Interpolator(4); fir
                y = fir(x)

                @savefig sdr_Interpolator_call_1.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.time_domain(x, marker="o", label="$x[n]$"); \
                sdr.plot.time_domain(y, sample_rate=fir.rate, marker=".", label="$y[n]$");

            Interpolate the signal using the `"full"` mode.

            .. ipython:: python

                y = fir(x, mode="full")

                @savefig sdr_Interpolator_call_2.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.time_domain(x, marker="o", label="$x[n]$"); \
                sdr.plot.time_domain(y, sample_rate=fir.rate, offset=-fir.delay/fir.rate, marker=".", label="$y[n]$");

            Interpolate the signal iteratively using the streaming mode.

            .. ipython:: python

                fir = sdr.Interpolator(4, streaming=True); fir

                y1 = fir(x[:10]); \
                y2 = fir(x[10:]); \
                y3 = fir(np.zeros(fir.polyphase_taps.shape[1])); \
                y = np.concatenate((y1, y2, y3))

                @savefig sdr_Interpolator_call_3.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.time_domain(x, marker="o", label="$x[n]$"); \
                sdr.plot.time_domain(y, sample_rate=fir.rate, offset=-fir.delay/fir.rate, marker=".", label="$y[n]$");
        """
        x = np.atleast_1d(x)
        if not x.ndim == 1:
            raise ValueError(f"Argument 'x' must be a 1-D, not {x.ndim}-D.")

        y, self._state = _polyphase_interpolate(x, self._state, self.polyphase_taps, mode, self.streaming)

        return y

    def __repr__(self) -> str:
        """
        Returns a code-styled string representation of the object.

        Examples:
            .. ipython:: python

                fir = sdr.Interpolator(7)
                fir
        """
        if self.method == "custom":
            h_str = np.array2string(self.taps, max_line_width=int(1e6), separator=", ", suppress_small=True)
        else:
            h_str = repr(self.method)
        return f"sdr.{type(self).__name__}({self.rate}, {h_str}, streaming={self.streaming})"

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the object.

        Examples:
            .. ipython:: python

                fir = sdr.Interpolator(7)
                print(fir)
        """
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  rate: {self.rate}"
        string += f"\n  method: {self._method!r}"
        string += f"\n  polyphase_taps: {self.polyphase_taps.shape} shape"
        string += f"\n  delay: {self.delay}"
        string += f"\n  streaming: {self.streaming}"
        return string

    ##############################################################################
    # Streaming mode
    ##############################################################################

    def reset(self):
        self._state = np.zeros(self.polyphase_taps.shape[1] - 1)

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def rate(self) -> int:
        """
        The interpolation rate $r$.
        """
        return self._rate

    @property
    def method(self) -> Literal["kaiser", "linear", "linear-matlab", "zoh", "custom"]:
        """
        The method used to design the multirate filter.
        """
        return self._method

    @property
    def taps(self) -> npt.NDArray:
        """
        The prototype feedforward taps $h_i$.

        Notes:
            The prototype feedforward taps $h_i$ are the feedforward taps of the FIR filter before
            polyphase decomposition. The polyphase feedforward taps $h_{i, j}$ are the feedforward taps
            of the FIR filter after polyphase decomposition.

            The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

            $$h_{i, j} = h_{i + j r} .$$

        Examples:
            .. ipython:: python

                fir = sdr.Interpolator(3, np.arange(10))
                fir.taps
                fir.polyphase_taps
        """
        return self._taps

    @property
    def polyphase_taps(self) -> npt.NDArray:
        """
        The polyphase feedforward taps $h_{i, j}$.

        Notes:
            The prototype feedforward taps $h_i$ are the feedforward taps of the FIR filter before
            polyphase decomposition. The polyphase feedforward taps $h_{i, j}$ are the feedforward taps
            of the FIR filter after polyphase decomposition.

            The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

            $$h_{i, j} = h_{i + j r} .$$

        Examples:
            .. ipython:: python

                fir = sdr.Interpolator(3, np.arange(10))
                fir.taps
                fir.polyphase_taps
        """
        return self._polyphase_taps

    @property
    def delay(self) -> int:
        """
        The delay of FIR filter in samples. The delay indicates the output sample index that corresponds to the
        first input sample.
        """
        return super().delay


def _polyphase_decimate(
    x: npt.NDArray,
    state: npt.NDArray,
    H: npt.NDArray,
    mode: Literal["rate", "full"],
    streaming: bool,
) -> tuple[npt.NDArray, npt.NDArray]:
    B, M = H.shape  # Number of polyphase branches, polyphase filter length
    dtype = np.result_type(x, H)

    if streaming:
        x_pad = np.concatenate((state, x))  # Prepend previous inputs from last __call__() call
        X_cols = x_pad.size // B
        X_pad = x_pad[0 : X_cols * B].reshape(X_cols, B).T  # Commutate across polyphase filters
        X_pad = np.flipud(X_pad)  # Commutate from bottom to top
        Y = np.zeros((B, X_cols - M + 1), dtype=dtype)
        for i in range(B):
            Y[i] = scipy.signal.convolve(X_pad[i], H[i], mode="valid")
        state = x_pad[-(B * M - 1) :]
    else:
        # Prepend zeros to so the first sample is alone in the first commutated column
        x_pad = np.insert(x, 0, np.zeros(B - 1))
        # Append zeros to input signal to distribute evenly across the B branches
        x_pad = np.append(x_pad, np.zeros(B - (x_pad.size % B), dtype=dtype))
        X_cols = x_pad.size // B
        X_pad = x_pad.reshape(X_cols, B).T  # Commutate across polyphase filters
        X_pad = np.flipud(X_pad)  # Commutate from bottom to top
        if mode == "full":
            Y = np.zeros((B, X_cols + M - 1), dtype=dtype)
            for i in range(B):
                Y[i] = scipy.signal.convolve(X_pad[i], H[i], mode="full")
        elif mode == "rate":
            Y = np.zeros((B, X_cols), dtype=dtype)
            for i in range(B):
                corr = scipy.signal.convolve(X_pad[i], H[i], mode="full")
                Y[i] = corr[M // 2 : M // 2 + X_cols]
        else:
            raise ValueError(f"Argument 'mode' must be 'rate' or 'full', not {mode!r}.")

    # Sum the outputs of the polyphase filters
    y = np.sum(Y, axis=0)

    return y, state


@export
class Decimator(FIR):
    r"""
    Implements a polyphase decimating FIR filter.

    Notes:
        The polyphase decimating filter is equivalent to first filtering the input signal $x[n]$ with the
        prototype FIR filter with feedforward coefficients $h_{i}$ and then decimating the filtered signal
        by $r$ (by discarding $r-1$ samples between each sample).

        Instead, the polyphase decimating filter first decomposes the prototype FIR filter into $r$ polyphase
        filters with feedforward coefficients $h_{i, j}$. The polyphase filters are then applied to the
        commutated input signal $x[n]$ in parallel. The outputs of the polyphase filters are then summed.
        This prevents the need to compute outputs that will be discarded, as is done in the first case.

        .. code-block:: text
           :caption: Polyphase 2x Decimating FIR Filter Block Diagram

                                     +------------------------+
            ..., x[4], x[2], x[0] -->| h[0], h[2], h[4], h[6] |--+
                                     +------------------------+  |
                                                                 @--> ..., y[1], y[0]
                                     +------------------------+  |
            ..., x[3], x[1], 0    -->| h[1], h[3], h[5], 0    |--+
                                     +------------------------+

            Input Commutator                                      Output Summation
            (bottom-to-top)

            x[n] = Input signal with sample rate fs
            y[n] = Output signal with sample rate fs/r
            h[n] = Prototype FIR filter
            @ = Adder

        The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

        $$h_{i, j} = h_{i + j r} .$$

    Examples:
        Create an input signal to interpolate.

        .. ipython:: python

            x = np.cos(np.pi / 64 * np.arange(280))

        Create a polyphase filter that decimates by 7 using the Kaiser window method.

        .. ipython:: python

            fir = sdr.Decimator(7); fir
            y = fir(x)

            @savefig sdr_Decimator_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, marker=".", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=1/fir.rate, marker="o", label="Decimated"); \
            plt.title("Decimation by 7 with the Kaiser window method"); \
            plt.tight_layout();

        Create a streaming polyphase filter that decimates by 7 using the Kaiser window method. This filter
        preserves state between calls.

        .. ipython:: python

            fir = sdr.Decimator(7, streaming=True); fir

            y1 = fir(x[0:70]); \
            y2 = fir(x[70:140]); \
            y3 = fir(x[140:210]); \
            y4 = fir(x[210:280]); \
            y5 = fir.flush()

            @savefig sdr_Decimator_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, marker=".", label="Input"); \
            sdr.plot.time_domain(y1, sample_rate=1/fir.rate, offset=-fir.delay*fir.rate + 0, marker="o", label="Decimated $y_1[n]$"); \
            sdr.plot.time_domain(y2, sample_rate=1/fir.rate, offset=-fir.delay*fir.rate + 70, marker="o", label="Decimated $y_2[n]$"); \
            sdr.plot.time_domain(y3, sample_rate=1/fir.rate, offset=-fir.delay*fir.rate + 140, marker="o", label="Decimated $y_3[n]$"); \
            sdr.plot.time_domain(y4, sample_rate=1/fir.rate, offset=-fir.delay*fir.rate + 210, marker="o", label="Decimated $y_4[n]$"); \
            sdr.plot.time_domain(y5, sample_rate=1/fir.rate, offset=-fir.delay*fir.rate + 280, marker="o", label="Decimated $y_5[n]$"); \
            plt.title("Streaming decimation by 7 with the Kaiser window method"); \
            plt.tight_layout();

    Group:
        dsp-multirate-filtering
    """

    def __init__(
        self,
        rate: int,
        taps: Literal["kaiser"] | npt.ArrayLike = "kaiser",
        streaming: bool = False,
    ):
        r"""
        Creates a polyphase FIR decimating filter.

        Arguments:
            rate: The decimation rate $r$.
            taps: The multirate filter design specification.

                - `"kaiser"`: The multirate filter is designed using :func:`~sdr.multirate_taps()`
                  with arguments 1 and `rate`.
                - `npt.ArrayLike`: The multirate filter feedforward coefficients $h_i$.

            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~Decimator.__call__()`.
        """
        if not isinstance(rate, int):
            raise TypeError("Argument 'rate' must be an integer.")
        if not rate >= 1:
            raise ValueError(f"Argument 'rate' must be at least 1, not {rate}.")
        self._rate = rate
        self._method: Literal["kaiser", "custom"]

        if not isinstance(taps, str):
            self._method = "custom"
            taps = np.asarray(taps)
        elif taps == "kaiser":
            self._method = "kaiser"
            taps = multirate_taps(1, rate)
        else:
            raise ValueError(f"Argument 'taps' must be 'kaiser', or an array-like, not {taps}.")

        # N = math.ceil(self.taps.size / rate) * rate
        # self._polyphase_taps = np.pad(self.taps, (0, N - self.taps.size), mode="constant").reshape(-1, rate).T
        self._polyphase_taps = polyphase_matrix(1, rate, taps)

        super().__init__(taps, streaming=streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike, mode: Literal["rate", "full"] = "rate") -> npt.NDArray:
        """
        Filters and decimates the input signal $x[n]$ with the polyphase FIR filter.

        Arguments:
            x: The input signal $x[n]$ with sample rate $f_s$ and length $L$.
            mode: The non-streaming convolution mode.

                - `"rate"`: The output signal $y[n]$ has length $L / r$ proportional to the decimation rate $r$.
                  Output sample 0 aligns with input sample 0.
                - `"full"`: The full convolution is performed. The output signal $y[n]$ has length $(L + N) r$,
                  where $N$ is the order of the multirate filter. Output sample :obj:`~sdr.Interpolator.delay` aligns
                  with input sample 0.

                In streaming mode, the `"full"` convolution is performed. However, for each $L$ input samples
                only $L / r$ output samples are produced per call. A final call with input zeros is required to flush
                the filter state.

        Returns:
            The filtered signal $y[n]$ with sample rate $f_s / r$. The output length is dictated by
            the `mode` argument.
        """
        x = np.atleast_1d(x)
        if not x.ndim == 1:
            raise ValueError(f"Argument 'x' must be a 1-D, not {x.ndim}-D.")

        y, self._state = _polyphase_decimate(x, self._state, self.polyphase_taps, mode, self.streaming)

        return y

    def __repr__(self) -> str:
        """
        Returns a code-styled string representation of the object.

        Examples:
            .. ipython:: python

                fir = sdr.Decimator(7)
                fir
        """
        if self.method == "custom":
            h_str = np.array2string(self.taps, max_line_width=int(1e6), separator=", ", suppress_small=True)
        else:
            h_str = repr(self.method)
        return f"sdr.{type(self).__name__}({self.rate}, {h_str}, streaming={self.streaming})"

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the object.

        Examples:
            .. ipython:: python

                fir = sdr.Decimator(7)
                print(fir)
        """
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  rate: {self.rate}"
        string += f"\n  method: {self._method!r}"
        string += f"\n  polyphase_taps: {self.polyphase_taps.shape} shape"
        string += f"\n  delay: {self.delay}"
        string += f"\n  streaming: {self.streaming}"
        return string

    ##############################################################################
    # Streaming mode
    ##############################################################################

    def reset(self):
        self._state = np.zeros(self.polyphase_taps.size - 1)

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def rate(self) -> int:
        """
        The decimation rate $r$.
        """
        return self._rate

    @property
    def method(self) -> Literal["kaiser", "custom"]:
        """
        The method used to design the multirate filter.
        """
        return self._method

    @property
    def taps(self) -> npt.NDArray:
        """
        The prototype feedforward taps $h_i$.

        Notes:
            The prototype feedforward taps $h_i$ are the feedforward taps of the FIR filter before
            polyphase decomposition. The polyphase feedforward taps $h_{i, j}$ are the feedforward taps
            of the FIR filter after polyphase decomposition.

            The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

            $$h_{i, j} = h_{i + j r} .$$

        Examples:
            .. ipython:: python

                fir = sdr.Decimator(3, np.arange(10))
                fir.taps
                fir.polyphase_taps
        """
        return self._taps

    @property
    def polyphase_taps(self) -> npt.NDArray:
        """
        The polyphase feedforward taps $h_{i, j}$.

        Notes:
            The prototype feedforward taps $h_i$ are the feedforward taps of the FIR filter before
            polyphase decomposition. The polyphase feedforward taps $h_{i, j}$ are the feedforward taps
            of the FIR filter after polyphase decomposition.

            The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

            $$h_{i, j} = h_{i + j r} .$$

        Examples:
            .. ipython:: python

                fir = sdr.Decimator(3, np.arange(10))
                fir.taps
                fir.polyphase_taps
        """
        return self._polyphase_taps

    @property
    def delay(self) -> int:
        """
        The delay of FIR filter in samples. The delay indicates the output sample index that corresponds to the
        first input sample.
        """
        assert super().delay % self.rate == 0, f"This should always be true. {super().delay} % {self.rate} != 0"
        return super().delay // self.rate


@export
class Resampler(FIR):
    r"""
    Implements a polyphase rational resampling FIR filter.

    Notes:
        The polyphase rational resampling filter is equivalent to first upsampling the input signal $x[n]$ by $P$
        (by inserting $P-1$ zeros between each sample), filtering the upsampled signal with the prototype FIR filter
        with feedforward coefficients $h_{i}$, and then decimating the filtered signal by $Q$ (by discarding $Q-1$
        samples between each sample).

        Instead, the polyphase rational resampling filter first decomposes the prototype FIR filter into $P$ polyphase
        filters with feedforward coefficients $h_{i, j}$. The polyphase filters are then applied to the
        input signal $x[n]$ in parallel. The output of the polyphase filters are then commutated by $Q$ to produce
        the output signal $y[n]$. This prevents the need to multiply with zeros in the upsampled input,
        as is needed in the first case.

        .. code-block:: text
           :caption: Polyphase 2/3 Resampling FIR Filter Block Diagram

                                  +------------------------+
                              +-->| h[0], h[2], h[4], h[6] |--> ..., ____, y[0]
                              |   +------------------------+
            ..., x[1], x[0] --+
                              |   +------------------------+
                              +-->| h[1], h[3], h[5], 0    |--> ..., y[1], ____
                                  +------------------------+

            Input Hold                                          Output Commutator by 3
                                                                (top-to-bottom)

            x[n] = Input signal with sample rate fs
            y[n] = Output signal with sample rate fs * P / Q
            h[n] = Prototype FIR filter

        The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

        $$h_{i, j} = h_{i + j P} .$$

        If the interpolation rate $P$ is 1, then the polyphase rational resampling filter is equivalent to the
        polyphase decimating filter. See :class:`~sdr.Decimator`.

    Examples:
        Create an input signal to resample.

        .. ipython:: python

            x = np.cos(np.pi / 4 * np.arange(40))

        Create a polyphase filter that resamples by 7/3 using the Kaiser window method.

        .. ipython:: python

            fir = sdr.Resampler(7, 3); fir
            y = fir(x)

            @savefig sdr_Resampler_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, marker=".", label="Resampled"); \
            plt.title("Resampling by 7/3 with the Kaiser window method"); \
            plt.tight_layout();

        Create a streaming polyphase filter that resamples by 7/3 using the Kaiser window method. This filter
        preserves state between calls.

        .. ipython:: python

            fir = sdr.Resampler(7, 3, streaming=True); fir

            y1 = fir(x[0:10]); \
            y2 = fir(x[10:20]); \
            y3 = fir(x[20:30]); \
            y4 = fir(x[30:40]); \
            y5 = fir.flush()

            @savefig sdr_Resampler_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y1, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 0, marker=".", label="Resampled $y_1[n]$"); \
            sdr.plot.time_domain(y2, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 10, marker=".", label="Resampled $y_2[n]$"); \
            sdr.plot.time_domain(y3, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 20, marker=".", label="Resampled $y_3[n]$"); \
            sdr.plot.time_domain(y4, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 30, marker=".", label="Resampled $y_4[n]$"); \
            sdr.plot.time_domain(y5, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 40, marker=".", label="Resampled $y_5[n]$"); \
            plt.title("Streaming resampling by 7/3 with the Kaiser window method"); \
            plt.tight_layout();

        Create a polyphase filter that resamples by 5/7 using linear method.

        .. ipython:: python

            fir = sdr.Resampler(5, 7); fir
            y = fir(x)

            @savefig sdr_Resampler_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, marker=".", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, marker="o", label="Resampled"); \
            plt.title("Resampling by 5/7 with the Kaiser window method"); \
            plt.tight_layout();

    Group:
        dsp-multirate-filtering
    """

    def __init__(
        self,
        up: int,
        down: int,
        taps: Literal["kaiser", "linear", "linear-matlab", "zoh"] | npt.ArrayLike = "kaiser",
        streaming: bool = False,
    ):
        r"""
        Creates a polyphase FIR rational resampling filter.

        Arguments:
            up: The interpolation rate $P$.
            down: The decimation rate $Q$.
            taps: The multirate filter design specification.

                - `"kaiser"`: The multirate filter is designed using :func:`~sdr.multirate_taps()`
                  with arguments `up` and `down`.
                - `"linear"`: The multirate filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2P$ linear ramp $\frac{1}{P} [0, ..., P-1, P, P-1, ..., 1]$.
                  The first output sample aligns with the first input sample.
                - `"linear-matlab"`: The multirate filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2P$ linear ramp $\frac{1}{P} [1, ..., P-1, P, P-1, ..., 0]$.
                  This is method MATLAB uses. The first output sample is advanced from the first input sample.
                - `"zoh"`: The multirate filter is designed to be a zero-order hold.
                  The filter coefficients are a length-$P$ array of ones.
                - `npt.ArrayLike`: The multirate filter feedforward coefficients $h_i$.

            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~Resampler.__call__()`.
        """
        if not isinstance(up, int):
            raise TypeError(f"Argument 'up' must be an integer, not {type(up)}.")
        if not up >= 1:
            raise ValueError(f"Argument 'up' must be at least 1, not {up}.")

        if not isinstance(down, int):
            raise TypeError(f"Argument 'down' must be an integer, not {type(down)}.")
        if not down >= 1:
            raise ValueError(f"Argument 'down' must be at least 1, not {down}.")

        self._up = up
        self._down = down
        self._rate = up / down
        self._method: Literal["kaiser", "linear", "linear-matlab", "zoh", "custom"]

        if not isinstance(taps, str):
            self._method = "custom"
            taps = np.asarray(taps)
        elif taps == "kaiser":
            self._method = "kaiser"
            taps = multirate_taps(up, down)
        elif taps == "linear":
            if not up > 1:
                raise ValueError(f"Argument 'up' must be greater than 1 to use the 'linear' method, not {up}.")
            self._method = "linear"
            taps = _multirate_taps_linear(up)
        elif taps == "linear-matlab":
            if not up > 1:
                raise ValueError(f"Argument 'up' must be greater than 1 to use the 'linear-matlab' method, not {up}.")
            self._method = "linear-matlab"
            taps = _multirate_taps_linear_matlab(up)
        elif taps == "zoh":
            if not up > 1:
                raise ValueError(f"Argument 'up' must be greater than 1 to use the 'zoh' method, not {up}.")
            self._method = "zoh"
            taps = _multirate_taps_zoh(up)
        else:
            raise ValueError(
                f"Argument 'taps' must be 'kaiser', 'linear', 'linear-matlab', 'zoh', or an array-like, not {taps}."
            )

        self._polyphase_taps = polyphase_matrix(up, down, taps)

        super().__init__(taps, streaming=streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike, mode: Literal["rate", "full"] = "rate") -> np.ndarray:
        r"""
        Resamples and filters the input signal $x[n]$ with the polyphase FIR filter.
        """
        x = np.atleast_1d(x)
        if not x.ndim == 1:
            raise ValueError(f"Argument 'x' must be a 1-D, not {x.ndim}-D.")

        if self.up == 1:
            y, self._state = _polyphase_decimate(x, self._state, self.polyphase_taps, mode, self.streaming)
        else:
            y, self._state = _polyphase_interpolate(x, self._state, self.polyphase_taps, mode, self.streaming)
            y = y[:: self.down]  # Downsample the interpolated output

        return y

    def __repr__(self) -> str:
        """
        Returns a code-styled string representation of the object.

        Examples:
            .. ipython:: python

                fir = sdr.Resampler(4, 3)
                fir
        """
        if self.method == "custom":
            h_str = np.array2string(self.taps, max_line_width=int(1e6), separator=", ", suppress_small=True)
        else:
            h_str = repr(self.method)
        return f"sdr.{type(self).__name__}({self.up}, {self.down}, {h_str}, streaming={self.streaming})"

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the object.

        Examples:
            .. ipython:: python

                fir = sdr.Resampler(4, 3)
                print(fir)
        """
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  rate: {self.up} / {self.down}"
        string += f"\n  method: {self._method!r}"
        string += f"\n  polyphase_taps: {self.polyphase_taps.shape} shape"
        string += f"\n  delay: {self.delay}"
        string += f"\n  streaming: {self.streaming}"
        return string

    ##############################################################################
    # Streaming mode
    ##############################################################################

    def reset(self):
        self._state = np.zeros(self.polyphase_taps.shape[1] - 1)

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def up(self) -> int:
        """
        The interpolation rate $P$.
        """
        return self._up

    @property
    def down(self) -> int:
        """
        The decimation rate $Q$.
        """
        return self._down

    @property
    def rate(self) -> float:
        """
        The resampling rate $P/Q$.
        """
        return self._rate

    @property
    def method(self) -> Literal["kaiser", "linear", "linear-matlab", "zoh", "custom"]:
        """
        The method used to design the multirate filter.
        """
        return self._method

    @property
    def taps(self) -> npt.NDArray:
        """
        The prototype feedforward taps $h_i$.

        Notes:
            The prototype feedforward taps $h_i$ are the feedforward taps of the FIR filter before
            polyphase decomposition. The polyphase feedforward taps $h_{i, j}$ are the feedforward taps
            of the FIR filter after polyphase decomposition.

            The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

            $$h_{i, j} = h_{i + j P} .$$

        Examples:
            .. ipython:: python

                fir = sdr.Resampler(4, 3, np.arange(10))
                fir.taps
                fir.polyphase_taps
        """
        return self._taps

    @property
    def polyphase_taps(self) -> npt.NDArray:
        """
        The polyphase feedforward taps $h_{i, j}$.

        Notes:
            The prototype feedforward taps $h_i$ are the feedforward taps of the FIR filter before
            polyphase decomposition. The polyphase feedforward taps $h_{i, j}$ are the feedforward taps
            of the FIR filter after polyphase decomposition.

            The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

            $$h_{i, j} = h_{i + j P} .$$

        Examples:
            .. ipython:: python

                fir = sdr.Resampler(4, 3, np.arange(10))
                fir.taps
                fir.polyphase_taps
        """
        return self._polyphase_taps

    @property
    def delay(self) -> int:
        """
        The delay of FIR filter in samples. The delay indicates the output sample index that corresponds to the
        first input sample.
        """
        assert super().delay % self.down == 0, f"This should always be true. {super().delay} % {self.down} != 0"
        return super().delay // self.down
