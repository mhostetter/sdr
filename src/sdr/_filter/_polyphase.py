"""
A module for interpolating finite impulse response (FIR) filters.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._helper import export
from ._design_multirate import (
    design_multirate_fir,
    design_multirate_fir_linear,
    design_multirate_fir_linear_matlab,
    design_multirate_fir_zoh,
    polyphase_decompose,
)
from ._fir import FIR


def _polyphase_input_hold(
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

    return Y, state


def _polyphase_input_commutate(
    x: npt.NDArray,
    state: npt.NDArray,
    H: npt.NDArray,
    mode: Literal["rate", "full"],
    bottom_to_top: bool,
    streaming: bool,
) -> tuple[npt.NDArray, npt.NDArray]:
    B, M = H.shape  # Number of polyphase branches, polyphase filter length
    dtype = np.result_type(x, H)

    if streaming:
        x_pad = np.concatenate((state, x))  # Prepend previous inputs from last __call__() call
        X_cols = x_pad.size // B
        X_pad = x_pad[0 : X_cols * B].reshape(X_cols, B).T  # Commutate across polyphase filters
        if bottom_to_top:
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
        if bottom_to_top:
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

    return Y, state


@export
class PolyphaseFIR(FIR):
    r"""
    Implements a generic polyphase FIR filter.

    Notes:
        The polyphase feedforward taps $h_i[n]$ are related to the prototype feedforward taps $h[n]$, given the
        number of polyphase branches $B$, by

        $$h_i[j] = h[i + j B] .$$

        The input signal $x[n]$ can be passed to each polyphase partition (used in interpolation) or commutated from
        bottom to top (used in decimation).

        .. md-tab-set::

            .. md-tab-item:: `input="hold"`

                .. code-block:: text
                   :caption: Polyphase FIR Filter with Input Hold

                                          +------------------------+
                                      +-->| h[0], h[3], h[6], h[9] |
                                      |   +------------------------+
                                      |   +------------------------+
                    ..., x[1], x[0] --+-->| h[1], h[4], h[7], 0    |
                                      |   +------------------------+
                                      |   +------------------------+
                                      +-->| h[2], h[5], h[8], 0    |
                                          +------------------------+

            .. md-tab-item:: `input="top-to-bottom"`

                .. code-block:: text
                   :caption: Polyphase FIR Filter with Input Commutation (Top to Bottom)

                                             +------------------------+
                    ..., x[4], x[1], 0    -->| h[0], h[3], h[6], h[9] |
                                             +------------------------+
                                             +------------------------+
                    ..., x[5], x[2], 0    -->| h[1], h[4], h[7], 0    |
                                             +------------------------+
                                             +------------------------+
                    ..., x[6], x[3], x[0] -->| h[2], h[5], h[8], 0    |
                                             +------------------------+

            .. md-tab-item:: `input="bottom-to-top"`

                .. code-block:: text
                   :caption: Polyphase FIR Filter with Input Commutation (Bottom to Top)

                                             +------------------------+
                    ..., x[6], x[3], x[0] -->| h[0], h[3], h[6], h[9] |
                                             +------------------------+
                                             +------------------------+
                    ..., x[5], x[2], 0    -->| h[1], h[4], h[7], 0    |
                                             +------------------------+
                                             +------------------------+
                    ..., x[4], x[1], 0    -->| h[2], h[5], h[8], 0    |
                                             +------------------------+

        The output of each polyphase partition can be summed to produce the output signal $y[n]$ (used in decimation),
        commutated from top to bottom (used in interpolation), or taken as parallel outputs $y_i[n]$ (used in
        channelization).

        .. md-tab-set::

            .. md-tab-item:: `output="sum"`

                .. code-block:: text
                   :caption: Polyphase FIR Filter with Output Summation

                    +------------------------+
                    | h[0], h[3], h[6], h[9] |--+
                    +------------------------+  |
                    +------------------------+  v
                    | h[1], h[4], h[7], 0    |--@--> ..., y[1], y[0]
                    +------------------------+  ^
                    +------------------------+  |
                    | h[2], h[5], h[8], 0    |--+
                    +------------------------+

            .. md-tab-item:: `output="top-to-bottom"`

                .. code-block:: text
                   :caption: Polyphase FIR Filter with Output Commutation (Top to Bottom)

                    +------------------------+
                    | h[0], h[3], h[6], h[9] |--> ..., y[3], y[0]
                    +------------------------+
                    +------------------------+
                    | h[1], h[4], h[7], 0    |--> ..., y[4], y[1]
                    +------------------------+
                    +------------------------+
                    | h[2], h[5], h[8], 0    |--> ..., y[5], y[2]
                    +------------------------+

            .. md-tab-item:: `output="bottom-to-top"`

                .. code-block:: text
                   :caption: Polyphase FIR Filter with Output Commutation (Bottom to Top)

                    +------------------------+
                    | h[0], h[3], h[6], h[9] |--> ..., y[5], y[2]
                    +------------------------+
                    +------------------------+
                    | h[1], h[4], h[7], 0    |--> ..., y[4], y[1]
                    +------------------------+
                    +------------------------+
                    | h[2], h[5], h[8], 0    |--> ..., y[3], y[0]
                    +------------------------+

            .. md-tab-item:: `output="all"`

                .. code-block:: text
                   :caption: Polyphase FIR Filter with Parallel Outputs

                    +------------------------+
                    | h[0], h[3], h[6], h[9] |--> ..., y[0,1], y[0,0]
                    +------------------------+
                    +------------------------+
                    | h[1], h[4], h[7], 0    |--> ..., y[1,1], y[1,0]
                    +------------------------+
                    +------------------------+
                    | h[2], h[5], h[8], 0    |--> ..., y[2,1], y[2,0]
                    +------------------------+

    Group:
        dsp-polyphase-filtering
    """

    def __init__(
        self,
        branches: int,
        taps: npt.ArrayLike,
        input: Literal["hold", "top-to-bottom", "bottom-to-top"] = "hold",
        output: Literal["sum", "top-to-bottom", "bottom-to-top", "all"] = "sum",
        streaming: bool = False,
    ):
        r"""
        Creates a polyphase FIR filter.

        Arguments:
            branches: The number of polyphase branches $B$.
            taps: The prototype filter feedforward coefficients $h[n]$.
            input: The input connection method.

                - `"hold"`: The input signal $x[n]$ is passed to each polyphase partition (used in interpolation).
                - `"top-to-bottom"`: The input signal $x[n]$ is commutated across the polyphase partitions from
                  top to bottom.
                - `"bottom-to-top"`: The input signal $x[n]$ is commutated across the polyphase partitions from
                  bottom to top (used in decimation).

            output: The output connection method.

                - `"sum"`: The output of each polyphase partition is summed to produce the output signal $y[n]$ (used
                  in decimation).
                - `"top-to-bottom"`: The output of each polyphase partition is commutated from top to bottom (used in
                  interpolation).
                - `"bottom-to-top"`: The output of each polyphase partition is commutated from bottom to top.
                - `"all"`: The outputs of each polyphase partition are used in parallel (used in channelization).

            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~PolyphaseFIR.__call__()`.
        """
        if not isinstance(branches, int):
            raise TypeError("Argument 'branches' must be an integer.")
        if not branches >= 1:
            raise ValueError(f"Argument 'branches' must be at least 1, not {branches}.")
        self._branches = branches

        self._taps = np.asarray(taps)
        self._polyphase_taps = polyphase_decompose(self.branches, self.taps)

        if not input in ["hold", "top-to-bottom", "bottom-to-top"]:
            raise ValueError(f"Argument 'input' must be 'hold', 'top-to-bottom', or 'bottom-to-top', not {input!r}.")
        self._input = input

        if not output in ["sum", "top-to-bottom", "bottom-to-top", "all"]:
            raise ValueError(
                f"Argument 'output' must be 'sum', 'top-to-bottom', 'bottom-to-top', or 'all', not {output!r}."
            )
        self._output = output

        super().__init__(taps, streaming=streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike, mode: Literal["rate", "full"] = "rate") -> npt.NDArray:
        r"""
        Filters the input signal $x[n]$ with the polyphase FIR filter.

        Arguments:
            x: The input signal $x[n]$.

        Returns:
            The filtered signal $y[n]$.
        """
        x = np.atleast_1d(x)
        if not x.ndim == 1:
            raise ValueError(f"Argument 'x' must be a 1-D, not {x.ndim}-D.")

        if self.input == "hold":
            Y, self._state = _polyphase_input_hold(x, self._state, self.polyphase_taps, mode, self.streaming)
        elif self.input == "top-to-bottom":
            Y, self._state = _polyphase_input_commutate(
                x, self._state, self.polyphase_taps, mode, False, self.streaming
            )
        elif self.input == "bottom-to-top":
            Y, self._state = _polyphase_input_commutate(x, self._state, self.polyphase_taps, mode, True, self.streaming)
        else:
            raise NotImplementedError(f"Input connection type {self.input!r} is not supported.")

        if self.output == "sum":
            y = np.sum(Y, axis=0)
        elif self.output == "bottom-to-top":
            y = np.flipud(Y).T.flatten()
        elif self.output == "top-to-bottom":
            y = Y.T.flatten()
        elif self.output == "all":
            y = Y
        else:
            raise NotImplementedError(f"Output connection type {self.output!r} is not supported.")

        return y

    def __repr__(self) -> str:
        if self.method == "custom":
            h_str = np.array2string(self.taps, max_line_width=int(1e6), separator=", ", suppress_small=True)
        else:
            h_str = repr(self.method)
        return f"sdr.{type(self).__name__}({self.branches}, {h_str}, streaming={self.streaming})"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  branches: {self.branches}"
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
    def branches(self) -> int:
        """
        The number of polyphase branches $B$.
        """
        return self._branches

    @property
    def taps(self) -> npt.NDArray:
        """
        The prototype feedforward taps $h[n]$.

        Notes:
            The prototype feedforward taps $h[n]$ are the feedforward taps of the FIR filter before
            polyphase decomposition. The polyphase feedforward taps $h_i[n]$ are the feedforward taps
            of the FIR filter after polyphase decomposition.

            The polyphase feedforward taps $h_i[n]$ are related to the prototype feedforward taps $h[n]$, given the
            number of polyphase branches $B$, by

            $$h_i[j] = h[i + j B] .$$

        Examples:
            .. ipython:: python

                fir = sdr.PolyphaseFIR(3, np.arange(10))
                fir.taps
                fir.polyphase_taps
        """
        return self._taps

    @property
    def polyphase_taps(self) -> npt.NDArray:
        """
        The polyphase feedforward taps $h_i[n]$.

        Notes:
            The prototype feedforward taps $h[n]$ are the feedforward taps of the FIR filter before
            polyphase decomposition. The polyphase feedforward taps $h_i[n]$ are the feedforward taps
            of the FIR filter after polyphase decomposition.

            The polyphase feedforward taps $h_i[n]$ are related to the prototype feedforward taps $h[n]$, given the
            number of polyphase branches $B$, by

            $$h_i[j] = h[i + j B] .$$

        Examples:
            .. ipython:: python

                fir = sdr.PolyphaseFIR(3, np.arange(10))
                fir.taps
                fir.polyphase_taps
        """
        return self._polyphase_taps

    @property
    def input(self) -> Literal["hold", "top-to-bottom", "bottom-to-top"]:
        """
        The input connection method.

        Notes:

            - `"hold"`: The input signal $x[n]$ is passed to each polyphase partition (used in interpolation).
            - `"top-to-bottom"`: The input signal $x[n]$ is commutated from top to bottom.
            - `"bottom-to-top"`: The input signal $x[n]$ is commutated from bottom to top (used in decimation).
        """
        return self._input

    @property
    def output(self) -> Literal["sum", "top-to-bottom", "bottom-to-top", "all"]:
        """
        The output connection method.

        Notes:

            - `"sum"`: The output of each polyphase partition is summed to produce the output signal $y[n]$ (used in
              decimation).
            - `"top-to-bottom"`: The output of each polyphase partition is commutated from top to bottom (used in
              interpolation).
            - `"bottom-to-top"`: The output of each polyphase partition is commutated from bottom to top.
            - `"all"`: The outputs of each polyphase partition are used in parallel (used in channelization).
        """
        return self._output

    @property
    def delay(self) -> int:
        """
        The delay of FIR filter in samples. The delay indicates the output sample index that corresponds to the
        first input sample.
        """
        return super().delay


@export
class Interpolator(PolyphaseFIR):
    r"""
    Implements a polyphase interpolating FIR filter.

    Notes:
        The polyphase interpolating filter is equivalent to first upsampling the input signal $x[n]$ by $r$
        (by inserting $r-1$ zeros between each sample) and then filtering the upsampled signal with the
        prototype FIR filter with feedforward coefficients $h[n]$.

        Instead, the polyphase interpolating filter first decomposes the prototype FIR filter into $r$ polyphase
        filters with feedforward coefficients $H_i$. The polyphase filters are then applied to the
        input signal $x[n]$ in parallel. The output of the polyphase filters are then commutated to produce
        the output signal $y[n]$. This prevents the need to multiply with zeros in the upsampled input,
        as is needed in the first case.

        .. code-block:: text
           :caption: Polyphase 3x Interpolating FIR Filter Block Diagram

                                  +------------------------+
                              +-->| h[0], h[3], h[6], h[9] |--> ..., y[3], y[0]
                              |   +------------------------+
                              |   +------------------------+
            ..., x[1], x[0] --+-->| h[1], h[4], h[7], 0    |--> ..., y[4], y[1]
                              |   +------------------------+
                              |   +------------------------+
                              +-->| h[2], h[5], h[8], 0    |--> ..., y[5], y[2]
                                  +------------------------+

            Input Hold                                          Output Commutator
                                                                (top-to-bottom)

            x[n] = Input signal with sample rate fs
            y[n] = Output signal with sample rate fs * r
            h[n] = Prototype FIR filter

        The polyphase feedforward taps $h_i[n]$ are related to the prototype feedforward taps $h[n]$ by

        $$h_i[j] = h[i + j r] .$$

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
        dsp-polyphase-filtering
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

                - `"kaiser"`: The multirate filter is designed using :func:`~sdr.design_multirate_fir()`
                  with arguments `rate` and 1.
                - `"linear"`: The multirate filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2r$ linear ramp $\frac{1}{r} [0, ..., r-1, r, r-1, ..., 1]$.
                  The first output sample aligns with the first input sample.
                - `"linear-matlab"`: The multirate filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2r$ linear ramp $\frac{1}{r} [1, ..., r-1, r, r-1, ..., 0]$.
                  This is method MATLAB uses. The first output sample is advanced from the first input sample.
                - `"zoh"`: The multirate filter is designed to be a zero-order hold.
                  The filter coefficients are a length-$r$ array of ones.
                - `npt.ArrayLike`: The prototype filter feedforward coefficients $h[n]$.

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
            taps = design_multirate_fir(rate, 1)
        elif taps == "linear":
            self._method = "linear"
            taps = design_multirate_fir_linear(rate)
        elif taps == "linear-matlab":
            self._method = "linear-matlab"
            taps = design_multirate_fir_linear_matlab(rate)
        elif taps == "zoh":
            self._method = "zoh"
            taps = design_multirate_fir_zoh(rate)
        else:
            raise ValueError(
                f"Argument 'taps' must be 'kaiser', 'linear', 'linear-matlab', 'zoh', or an array-like, not {taps}."
            )

        super().__init__(rate, taps, input="hold", output="top-to-bottom", streaming=streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike, mode: Literal["rate", "full"] = "rate") -> npt.NDArray:
        r"""
        Interpolates and filters the input signal $x[n]$ with the polyphase FIR filter.

        Arguments:
            x: The input signal $x[n]$ with sample rate $f_s$ and length $L$.
            mode: The non-streaming convolution mode.

                - `"rate"`: The output signal $y[n]$ has length $L \cdot r$, proportional to the interpolation rate
                  $r$. Output sample 0 aligns with input sample 0.
                - `"full"`: The full convolution is performed. The output signal $y[n]$ has length $(L + N) r$,
                  where $N$ is the order of the multirate filter. Output sample :obj:`~sdr.Interpolator.delay` aligns
                  with input sample 0.

                In streaming mode, the `"full"` convolution is performed. However, for each $L$ input samples
                only $L \cdot r$ output samples are produced per call. A final call to :meth:`~Interpolator.flush()`
                is required to flush the filter state.

        Returns:
            The filtered signal $y[n]$ with sample rate $f_s \cdot r$. The output length is dictated by
            the `mode` argument.
        """
        return super().__call__(x, mode)

    def __repr__(self) -> str:
        if self.method == "custom":
            h_str = np.array2string(self.taps, max_line_width=int(1e6), separator=", ", suppress_small=True)
        else:
            h_str = repr(self.method)
        return f"sdr.{type(self).__name__}({self.rate}, {h_str}, streaming={self.streaming})"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  rate: {self.rate}"
        string += f"\n  method: {self._method!r}"
        string += f"\n  polyphase_taps: {self.polyphase_taps.shape} shape"
        string += f"\n  delay: {self.delay}"
        string += f"\n  streaming: {self.streaming}"
        return string

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
    def delay(self) -> int:
        """
        The delay of FIR filter in samples. The delay indicates the output sample index that corresponds to the
        first input sample.
        """
        return super().delay


@export
class Decimator(PolyphaseFIR):
    r"""
    Implements a polyphase decimating FIR filter.

    Notes:
        The polyphase decimating filter is equivalent to first filtering the input signal $x[n]$ with the
        prototype FIR filter with feedforward coefficients $h[n]$ and then decimating the filtered signal
        by $r$ (by discarding $r-1$ samples between each sample).

        Instead, the polyphase decimating filter first decomposes the prototype FIR filter into $r$ polyphase
        filters with feedforward coefficients $h_i[n]$. The polyphase filters are then applied to the
        commutated input signal $x[n]$ in parallel. The outputs of the polyphase filters are then summed.
        This prevents the need to compute outputs that will be discarded, as is done in the first case.

        .. code-block:: text
           :caption: Polyphase 3x Decimating FIR Filter Block Diagram

                                     +------------------------+
            ..., x[6], x[3], x[0] -->| h[0], h[3], h[6], h[9] |---+
                                     +------------------------+   |
                                     +------------------------+   v
            ..., x[5], x[2], 0    -->| h[1], h[4], h[7], 0    |-->@--> ..., y[1], y[0]
                                     +------------------------+   ^
                                     +------------------------+   |
            ..., x[4], x[1], 0    -->| h[2], h[5], h[8], 0    |---+
                                     +------------------------+

            Input Commutator                                           Output Summation
            (bottom-to-top)

            x[n] = Input signal with sample rate fs
            y[n] = Output signal with sample rate fs/r
            h[n] = Prototype FIR filter
            @ = Adder

        The polyphase feedforward taps $h_i[n]$ are related to the prototype feedforward taps $h[n]$ by

        $$h_i[j] = h[i + j r] .$$

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
        dsp-polyphase-filtering
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

                - `"kaiser"`: The multirate filter is designed using :func:`~sdr.design_multirate_fir()`
                  with arguments 1 and `rate`.
                - `npt.ArrayLike`: The prototype filter feedforward coefficients $h[n]$.

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
            taps = design_multirate_fir(1, rate)
        else:
            raise ValueError(f"Argument 'taps' must be 'kaiser', or an array-like, not {taps!r}.")

        super().__init__(rate, taps, input="bottom-to-top", output="sum", streaming=streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike, mode: Literal["rate", "full"] = "rate") -> npt.NDArray:
        r"""
        Filters and decimates the input signal $x[n]$ with the polyphase FIR filter.

        Arguments:
            x: The input signal $x[n]$ with sample rate $f_s$ and length $L$.
            mode: The non-streaming convolution mode.

                - `"rate"`: The output signal $y[n]$ has length $L / r$, proportional to the decimation rate $r$.
                  Output sample 0 aligns with input sample 0.
                - `"full"`: The full convolution is performed. The output signal $y[n]$ has length $(L + N) r$,
                  where $N$ is the order of the multirate filter. Output sample :obj:`~sdr.Decimator.delay` aligns
                  with input sample 0.

                In streaming mode, the `"full"` convolution is performed. However, for each $L$ input samples
                only $L / r$ output samples are produced per call. A final call to :meth:`~Decimator.flush()`
                is required to flush the filter state.

        Returns:
            The filtered signal $y[n]$ with sample rate $f_s / r$. The output length is dictated by
            the `mode` argument.
        """
        return super().__call__(x, mode)

    def __repr__(self) -> str:
        if self.method == "custom":
            h_str = np.array2string(self.taps, max_line_width=int(1e6), separator=", ", suppress_small=True)
        else:
            h_str = repr(self.method)
        return f"sdr.{type(self).__name__}({self.rate}, {h_str}, streaming={self.streaming})"

    def __str__(self) -> str:
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
    def delay(self) -> int:
        """
        The delay of FIR filter in samples. The delay indicates the output sample index that corresponds to the
        first input sample.
        """
        assert super().delay % self.rate == 0, f"This should always be true. {super().delay} % {self.rate} != 0"
        return super().delay // self.rate


@export
class Resampler(PolyphaseFIR):
    r"""
    Implements a polyphase rational resampling FIR filter.

    Notes:
        The polyphase rational resampling filter is equivalent to first upsampling the input signal $x[n]$ by $P$
        (by inserting $P-1$ zeros between each sample), filtering the upsampled signal with the prototype FIR filter
        with feedforward coefficients $h[n]$, and then decimating the filtered signal by $Q$ (by discarding $Q-1$
        samples between each sample).

        Instead, the polyphase rational resampling filter first decomposes the prototype FIR filter into $P$ polyphase
        filters with feedforward coefficients $h_i[n]$. The polyphase filters are then applied to the
        input signal $x[n]$ in parallel. The output of the polyphase filters are then commutated by $Q$ to produce
        the output signal $y[n]$. This prevents the need to multiply with zeros in the upsampled input,
        as is needed in the first case.

        .. code-block:: text
           :caption: Polyphase 3/2 Resampling FIR Filter Block Diagram

                                  +------------------------+
                              +-->| h[0], h[3], h[6], h[9] |--> ..., ____, y[0]
                              |   +------------------------+
                              |   +------------------------+
            ..., x[1], x[0] --+-->| h[1], h[4], h[7], 0    |--> ..., y[2], ____
                              |   +------------------------+
                              |   +------------------------+
                              +-->| h[2], h[5], h[8], 0    |--> ..., ____, y[1]
                                  +------------------------+

            Input Hold                                          Output Commutator by 2
                                                                (top-to-bottom)

            x[n] = Input signal with sample rate fs
            y[n] = Output signal with sample rate fs * P / Q
            h[n] = Prototype FIR filter

        The polyphase feedforward taps $h_i[n]$ are related to the prototype feedforward taps $h[n]$ by

        $$h_i[j] = h[i + j P] .$$

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
        dsp-polyphase-filtering
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

                - `"kaiser"`: The multirate filter is designed using :func:`~sdr.design_multirate_fir()`
                  with arguments `up` and `down`.
                - `"linear"`: The multirate filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2P$ linear ramp $\frac{1}{P} [0, ..., P-1, P, P-1, ..., 1]$.
                  The first output sample aligns with the first input sample.
                - `"linear-matlab"`: The multirate filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2P$ linear ramp $\frac{1}{P} [1, ..., P-1, P, P-1, ..., 0]$.
                  This is method MATLAB uses. The first output sample is advanced from the first input sample.
                - `"zoh"`: The multirate filter is designed to be a zero-order hold.
                  The filter coefficients are a length-$P$ array of ones.
                - `npt.ArrayLike`: The prototype filter feedforward coefficients $h[n]$.

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
            taps = design_multirate_fir(up, down)
        elif taps == "linear":
            if not up > 1:
                raise ValueError(f"Argument 'up' must be greater than 1 to use the 'linear' method, not {up}.")
            self._method = "linear"
            taps = design_multirate_fir_linear(up)
        elif taps == "linear-matlab":
            if not up > 1:
                raise ValueError(f"Argument 'up' must be greater than 1 to use the 'linear-matlab' method, not {up}.")
            self._method = "linear-matlab"
            taps = design_multirate_fir_linear_matlab(up)
        elif taps == "zoh":
            if not up > 1:
                raise ValueError(f"Argument 'up' must be greater than 1 to use the 'zoh' method, not {up}.")
            self._method = "zoh"
            taps = design_multirate_fir_zoh(up)
        else:
            raise ValueError(
                f"Argument 'taps' must be 'kaiser', 'linear', 'linear-matlab', 'zoh', or an array-like, not {taps!r}."
            )

        branches = up if up > 1 else down
        super().__init__(branches, taps, input="hold", output="top-to-bottom", streaming=streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike, mode: Literal["rate", "full"] = "rate") -> np.ndarray:
        r"""
        Resamples and filters the input signal $x[n]$ with the polyphase FIR filter.

        Arguments:
            x: The input signal $x[n]$ with sample rate $f_s$ and length $L$.
            mode: The non-streaming convolution mode.

                - `"rate"`: The output signal $y[n]$ has length $L \cdot P/Q$, proportional to the resampling rate
                  $P / Q$. Output sample 0 aligns with input sample 0.
                - `"full"`: The full convolution is performed. The output signal $y[n]$ has length $(L + N) P/Q$,
                  where $N$ is the order of the multirate filter. Output sample :obj:`~sdr.Resampler.delay` aligns
                  with input sample 0.

                In streaming mode, the `"full"` convolution is performed. However, for each $L$ input samples
                only $L \cdot P/Q$ output samples are produced per call. A final call to :meth:`~Resampler.flush()`
                is required to flush the filter state.

        Returns:
            The filtered signal $y[n]$ with sample rate $f_s \cdot P/Q$. The output length is dictated by
            the `mode` argument.
        """
        y = super().__call__(x, mode)

        if self.up > 1:
            y = y[:: self.down]  # Downsample the interpolated output

        return y

    def __repr__(self) -> str:
        if self.method == "custom":
            h_str = np.array2string(self.taps, max_line_width=int(1e6), separator=", ", suppress_small=True)
        else:
            h_str = repr(self.method)
        return f"sdr.{type(self).__name__}({self.up}, {self.down}, {h_str}, streaming={self.streaming})"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  rate: {self.up} / {self.down}"
        string += f"\n  method: {self._method!r}"
        string += f"\n  polyphase_taps: {self.polyphase_taps.shape} shape"
        string += f"\n  delay: {self.delay}"
        string += f"\n  streaming: {self.streaming}"
        return string

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
    def delay(self) -> int:
        """
        The delay of FIR filter in samples. The delay indicates the output sample index that corresponds to the
        first input sample.
        """
        assert super().delay % self.down == 0, f"This should always be true. {super().delay} % {self.down} != 0"
        return super().delay // self.down
