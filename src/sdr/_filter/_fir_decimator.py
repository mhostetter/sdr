"""
A module for decimating finite impulse response (FIR) filters.
"""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._helper import export
from ._fir_interpolator import FIR, multirate_taps, polyphase_matrix


@export
class Decimator(FIR):
    r"""
    Implements a polyphase finite impulse response (FIR) decimating filter.

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
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=1/fir.rate, marker=".", label="Decimated"); \
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
            sdr.plot.time_domain(y1, sample_rate=1/fir.rate, offset=-fir.delay + 0, marker=".", label="Decimated $y_1[n]$"); \
            sdr.plot.time_domain(y2, sample_rate=1/fir.rate, offset=-fir.delay + 70, marker=".", label="Decimated $y_2[n]$"); \
            sdr.plot.time_domain(y3, sample_rate=1/fir.rate, offset=-fir.delay + 140, marker=".", label="Decimated $y_3[n]$"); \
            sdr.plot.time_domain(y4, sample_rate=1/fir.rate, offset=-fir.delay + 210, marker=".", label="Decimated $y_4[n]$"); \
            sdr.plot.time_domain(y5, sample_rate=1/fir.rate, offset=-fir.delay + 280, marker=".", label="Decimated $y_5[n]$"); \
            plt.title("Streaming decimation by 7 with the Kaiser window method"); \
            plt.tight_layout();

    Group:
        dsp-multirate-filtering
    """  # pylint: disable=line-too-long

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

    def __call__(self, x: npt.ArrayLike, mode: Literal["rate", "full"] = "rate") -> np.ndarray:
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

        dtype = np.result_type(x, self.polyphase_taps)
        H = self.polyphase_taps
        B, M = self.polyphase_taps.shape  # Number of polyphase branches, polyphase filter length

        if self.streaming:
            x_pad = np.concatenate((self._state, x))  # Prepend previous inputs from last __call__() call
            X_cols = x_pad.size // B
            X_pad = x_pad[0 : X_cols * B].reshape(X_cols, B).T  # Commutate across polyphase filters
            X_pad = np.flipud(X_pad)  # Commutate from bottom to top
            Y = np.zeros((B, X_cols - M + 1), dtype=dtype)
            for i in range(B):
                Y[i] = scipy.signal.convolve(X_pad[i], H[i], mode="valid")
            self._state = x_pad[-(B * M - 1) :]
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
            h_str = np.array2string(self.taps, max_line_width=1e6, separator=", ", suppress_small=True)
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

                fir = sdr.Decimator(3, np.arange(10))
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

                fir = sdr.Decimator(3, np.arange(10))
                fir.taps
                fir.polyphase_taps
        """
        return self._polyphase_taps
