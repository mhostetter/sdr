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
def multirate_fir(
    P: int,
    Q: int = 1,
    half_length: int = 12,
    A_stop: float = 80,
) -> np.ndarray:
    r"""
    Computes the multirate FIR filter that achieves rational resampling by $P/Q$.

    Note:
        This filter can be used with :class:`sdr.FIRInterpolator` or :class:`sdr.FIRDecimator`.

    Arguments:
        P: The interpolation rate $P$.
        Q: The decimation rate $Q$.
        half_length: The half-length of the polyphase filters.
        A_stop: The stopband attenuation $A_{\text{stop}}$ in dB.

    Returns:
        The multirate FIR filter $h[n]$.

    References:
        - fred harris, *Multirate Signal Processing for Communication Systems*, Chapter 7: Resampling Filters
        - https://www.mathworks.com/help/dsp/ref/designmultiratefir.html

    Examples:
        Design a multirate FIR filter for rational resampling by 2/3.

        .. ipython:: python

            h = sdr.multirate_fir(2, 3)

            @savefig sdr_multirate_fir_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.impulse_response(h);

            @savefig sdr_multirate_fir_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.frequency_response(h);

    Group:
        dsp-multirate-filtering
    """
    if not isinstance(P, int):
        raise TypeError(f"Argument 'P' must be an integer, not {P}.")
    if not P >= 1:
        raise ValueError(f"Argument 'P' must be at least 1, not {P}.")

    if not isinstance(Q, int):
        raise TypeError(f"Argument 'Q' must be an integer, not {Q}.")
    if not Q >= 1:
        raise ValueError(f"Argument 'Q' must be at least 1, not {Q}.")

    B = P if P > 1 else Q  # The number of polyphase branches
    R = max(P, Q)  # Inverse of the filter's fractional bandwidth

    # Compute the filter order, which is length - 1
    N = 2 * half_length * B

    # Compute ideal lowpass filter
    n = np.arange(N + 1)
    h = P / R * np.sinc((n - N // 2) / R)

    # Compute Kaiser window
    if A_stop >= 50:
        beta = 0.1102 * (A_stop - 8.71)
    elif A_stop > 21:
        beta = 0.5842 * (A_stop - 21) ** 0.4 + 0.07886 * (A_stop - 21)
    else:
        beta = 0
    w = scipy.signal.kaiser(N + 1, beta)

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


@export
class FIRInterpolator(FIR):
    r"""
    Implements a polyphase finite impulse response (FIR) interpolating filter.

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

        The polyphase feedforward taps $h_{i, j}$ are related to the prototype feedforward taps $h_i$ by

        $$h_{i, j} = h_{i + j r} .$$

    Examples:
        Create an input signal to interpolate.

        .. ipython:: python

            x = np.cos(np.pi / 4 * np.arange(40))

        Create a polyphase filter that interpolates by 7 using the Kaiser window method.

        .. ipython:: python

            fir = sdr.FIRInterpolator(7); fir
            y = fir(x)

            @savefig sdr_fir_interpolator_1.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, offset=-fir.delay/fir.rate, marker=".", label="Filtered"); \
            plt.title("Interpolation by 7 with the Kaiser window method"); \
            plt.tight_layout();

        Create a polyphase filter that interpolates by 7 using linear method.

        .. ipython:: python

            fir = sdr.FIRInterpolator(7, "linear"); fir
            fir.polyphase_taps
            y = fir(x)

            @savefig sdr_fir_interpolator_2.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, offset=-fir.delay/fir.rate, marker=".", label="Filtered");
            plt.title("Interpolation by 7 with the linear method"); \
            plt.tight_layout();

        Create a polyphase filter that interpolates by 7 using the zero-order hold method.

        .. ipython:: python

            fir = sdr.FIRInterpolator(7, "zoh"); fir
            fir.polyphase_taps
            y = fir(x)

            @savefig sdr_fir_interpolator_3.png
            plt.figure(figsize=(8, 4)); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, offset=-fir.delay/fir.rate, marker=".", label="Filtered");
            plt.title("Interpolation by 7 with the zero-order hold method"); \
            plt.tight_layout();

    Group:
        dsp-multirate-filtering
    """

    def __init__(
        self,
        rate: int,
        taps: Literal["kaiser", "linear", "zoh"] | npt.ArrayLike = "kaiser",
        streaming: bool = False,
    ):
        r"""
        Creates a polyphase FIR interpolating filter with feedforward coefficients $h_i$.

        Arguments:
            rate: The interpolation rate $r$.
            taps: The multirate filter design specification.

                - `"kaiser"`: The multirate filter is designed using :func:`~sdr.multirate_fir()`
                  with arguments `rate` and 1.
                - `"linear"`: The multirate filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2r$ linear ramp $\frac{1}{r} [0, ..., r-1, r, r-1, ..., 1]$.
                - `"zoh"`: The multirate filter is designed to be a zero-order hold.
                  The filter coefficients are a length-$r$ array of ones.
                - `npt.ArrayLike`: The multirate filter feedforward coefficients $h_i$.

            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~FIRInterpolator.__call__()`.
        """
        if not isinstance(rate, int):
            raise TypeError("Argument 'rate' must be an integer.")
        if not rate >= 1:
            raise ValueError(f"Argument 'rate' must be at least 1, not {rate}.")
        self._rate = rate

        if not isinstance(taps, str):
            self._method = "custom"
            taps = np.asarray(taps)
        elif taps == "kaiser":
            self._method = "kaiser"
            taps = multirate_fir(rate, 1)
        elif taps == "linear":
            self._method = "linear"
            taps = np.zeros(2 * rate, dtype=np.float32)
            taps[:rate] = np.arange(0, rate) / rate
            taps[rate:] = np.arange(rate, 0, -1) / rate
        elif taps == "zoh":
            self._method = "zoh"
            taps = np.ones(rate, dtype=np.float32)
        else:
            raise ValueError(f"Argument 'taps' must be 'kaiser', 'linear', 'zoh', or an array-like, not {taps}.")

        N = math.ceil(taps.size / rate) * rate
        self._polyphase_taps = np.pad(taps, (0, N - taps.size), mode="constant").reshape(-1, rate).T

        super().__init__(taps, streaming=streaming)

    def __repr__(self) -> str:
        """
        Returns a code-styled string representation of the object.

        Examples:
            .. ipython:: python

                fir = sdr.FIRInterpolator(7)
                fir
        """
        if self.method == "custom":
            h_str = np.array2string(self.taps, max_line_width=1e6, separator=", ", suppress_small=True)
        else:
            h_str = self.method
        return f"sdr.{type(self).__name__}({self.rate}, {h_str}, streaming={self.streaming})"

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the object.

        Examples:
            .. ipython:: python

                fir = sdr.FIRInterpolator(7)
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

    def reset(self):
        self._x_prev = np.zeros(self.polyphase_taps.shape[1] - 1)

    def __call__(self, x: npt.ArrayLike, mode: Literal["full", "valid", "same"] = "full") -> np.ndarray:
        """
        Filters and interpolates the input signal $x[n]$ with the FIR filter.

        Arguments:
            x: The input signal $x[n]$ with sample rate $f_s$.
            mode: The non-streaming convolution mode. See :func:`scipy.signal.convolve` for details.
                In streaming mode, $N$ inputs always produce $N r$ outputs.

        Returns:
            The filtered signal $y[n]$ with sample rate $f_s r$.
        """
        x = np.atleast_1d(x)
        dtype = np.result_type(x, self.polyphase_taps)

        if self.streaming:
            # Prepend previous inputs from last __call__() call
            x_pad = np.concatenate((self._x_prev, x))
            yy = np.zeros((self.rate, x.size), dtype=dtype)
            for i in range(self.rate):
                yy[i] = scipy.signal.convolve(x_pad, self.polyphase_taps[i], mode="valid")

            self._x_prev = x_pad[-(self.polyphase_taps.shape[1] - 1) :]

            # xx = np.tile(x_pad, (self.rate, 1))
            # taps = np.tile(self.polyphase_taps, self.rate)
            # yy = scipy.signal.convolve(xx, taps, mode="valid")

            # Commutate the outputs of the polyphase filters
            y = yy.T.flatten()
        else:
            yy = np.zeros((self.rate, x.size + self.polyphase_taps.shape[1] - 1), dtype=dtype)
            for i in range(self.rate):
                yy[i] = scipy.signal.convolve(x, self.polyphase_taps[i], mode="full")

            # Commutate the outputs of the polyphase filters
            y = yy.T.flatten()

            if mode == "full":
                size = x.size * self.rate + self.taps.size - 1
                y = y[:size]
            elif mode == "same":
                size = max(x.size * self.rate, self.taps.size)
                offset = (min(x.size * self.rate, self.taps.size) - 1) // 2
                y = y[offset : offset + size]
            else:
                size = (max(x.size * self.rate, self.taps.size) - min(x.size * self.rate, self.taps.size)) + 1
                offset = min(x.size * self.rate, self.taps.size) - 1
                y = y[offset : offset + size]

            # xx = np.tile(x, (self.rate, 1))
            # taps = np.tile(self.polyphase_taps, self.rate)
            # yy = scipy.signal.convolve(xx, taps, mode="full")

        return y

    @property
    def method(self) -> Literal["kaiser", "linear", "zoh", "custom"]:
        """
        The method used to design the multirate filter.
        """
        return self._method

    @property
    def taps(self) -> np.ndarray:
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

                fir = sdr.FIRInterpolator(3, np.arange(10))
                fir.taps
                fir.polyphase_taps
        """
        return self._taps

    @property
    def polyphase_taps(self) -> np.ndarray:
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

                fir = sdr.FIRInterpolator(3, np.arange(10))
                fir.taps
                fir.polyphase_taps
        """
        return self._polyphase_taps

    @property
    def rate(self) -> int:
        """
        The interpolation rate $r$.
        """
        return self._rate
