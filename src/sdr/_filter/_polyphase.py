"""
A module for interpolating finite impulse response (FIR) filters.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._helper import convert_output, export, verify_arraylike, verify_bool, verify_literal, verify_scalar
from ._design_multirate import (
    multirate_fir,
    multirate_fir_linear,
    multirate_fir_linear_matlab,
    multirate_fir_zoh,
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
        self._branches = verify_scalar(branches, int=True, positive=True)
        self._taps = verify_arraylike(taps, ndim=1)
        self._input = verify_literal(input, ["hold", "top-to-bottom", "bottom-to-top"])
        self._output = verify_literal(output, ["sum", "top-to-bottom", "bottom-to-top", "all"])
        verify_bool(streaming)

        self._polyphase_taps = polyphase_decompose(self.branches, self.taps)

        # Determine the effective interpolation and decimation rates based on the input and output connection types
        self._interpolation = 1
        self._decimation = 1
        if self.input in ["top-to-bottom", "bottom-to-top"]:
            self._decimation *= self.branches
        if self.output in ["top-to-bottom", "bottom-to-top"]:
            self._interpolation *= self.branches

        super().__init__(taps, streaming=streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(
        self,
        x: npt.ArrayLike,
        mode: Literal["rate", "full"] = "rate",
    ) -> npt.NDArray:
        r"""
        Filters the input signal $x[n]$ with the polyphase FIR filter.

        Arguments:
            x: The input signal $x[n]$ with sample rate $f_s$ and length $L$.
            mode: The non-streaming convolution mode.

                - `"rate"`: The output signal $y[n]$ has length $L \cdot r$, proportional to the resampling rate
                  $r$. Output sample 0 aligns with input sample 0.
                - `"full"`: The full convolution is performed. The output signal $y[n]$ has length $(L + N) \cdot r$,
                  where $N$ is the order of the prototype filter. Output sample :obj:`~PolyphaseFIR.delay` aligns
                  with input sample 0.

                In streaming mode, the `"full"` convolution is performed. However, for each $L$ input samples
                only $L \cdot r$ output samples are produced per call. A final call to :meth:`~PolyphaseFIR.flush()`
                is required to flush the filter state.

        Returns:
            The filtered signal $y[n]$ with sample rate $f_s \cdot r$. The output length is dictated by
            the `mode` argument.
        """
        x = verify_arraylike(x, atleast_1d=True, ndim=1)
        verify_literal(mode, ["rate", "full"])

        if self.input == "hold":
            Y, self._state = _polyphase_input_hold(x, self._state, self.polyphase_taps, mode, self.streaming)
        elif self.input == "top-to-bottom":
            Y, self._state = _polyphase_input_commutate(
                x, self._state, self.polyphase_taps, mode, False, self.streaming
            )
        elif self.input == "bottom-to-top":
            Y, self._state = _polyphase_input_commutate(x, self._state, self.polyphase_taps, mode, True, self.streaming)

        if self.output == "sum":
            y = np.sum(Y, axis=0)
        elif self.output == "bottom-to-top":
            y = np.flipud(Y).T.flatten()
        elif self.output == "top-to-bottom":
            y = Y.T.flatten()
        elif self.output == "all":
            y = Y

        return convert_output(y)

    def __repr__(self) -> str:
        h_str = np.array2string(self.taps, max_line_width=int(1e6), separator=", ", suppress_small=True)
        return f"sdr.{type(self).__name__}({self.branches}, {h_str}, streaming={self.streaming})"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  branches: {self.branches}"
        string += f"\n  rate: {self.interpolation} / {self.decimation}"
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
    def order(self) -> int:
        """
        The order $N = (M + 1)B - 1$ of the FIR prototype filter $h[n]$.

        Examples:
            .. ipython:: python

                fir = sdr.PolyphaseFIR(3, np.arange(10))
                fir.taps
                fir.polyphase_taps
                fir.order
                fir.polyphase_order
        """
        return self.taps.size - 1

    @property
    def polyphase_order(self) -> int:
        """
        The order $M = (N + 1)/B - 1$ of each FIR polyphase filter $h_i[n]$.

        Examples:
            .. ipython:: python

                fir = sdr.PolyphaseFIR(3, np.arange(10))
                fir.taps
                fir.polyphase_taps
                fir.order
                fir.polyphase_order
        """
        return self.polyphase_taps.shape[1] - 1

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
    def interpolation(self) -> int:
        """
        The integer interpolation rate $P$.
        """
        return self._interpolation

    @property
    def decimation(self) -> int:
        """
        The integer decimation rate $Q$.
        """
        return self._decimation

    @property
    def rate(self) -> float:
        r"""
        The fractional resampling rate $r = P/Q$. The output sample rate is $f_{s,out} = f_{s,in} \cdot r$.
        """
        return self.interpolation / self.decimation

    @property
    def delay(self) -> int:
        """
        The delay of polyphase FIR filter in samples.

        The delay indicates the output sample index that corresponds to the first input sample.
        """
        return super().delay // self.decimation


@export
class Interpolator(PolyphaseFIR):
    r"""
    Implements a polyphase interpolating FIR filter.

    Notes:
        The polyphase interpolating filter is equivalent to first upsampling the input signal $x[n]$ by $P$
        (by inserting $P-1$ zeros between each sample) and then filtering the upsampled signal with the
        prototype FIR filter with feedforward coefficients $h[n]$.

        Instead, the polyphase interpolating filter first decomposes the prototype FIR filter into $P$ polyphase
        filters with feedforward coefficients $h_i[n]$. The polyphase filters are then applied to the
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
            y[n] = Output signal with sample rate fs * P
            h[n] = Prototype FIR filter

        The polyphase feedforward taps $h_i[n]$ are related to the prototype feedforward taps $h[n]$ by

        $$h_i[j] = h[i + j P] .$$

    References:
        - fred harris, *Multirate Signal Processing for Communication Systems*, Chapter 7: Resampling Filters.

    Examples:
        Create an input signal to interpolate.

        .. ipython:: python

            x = np.cos(np.pi / 4 * np.arange(40))

        Create a polyphase filter that interpolates by 7 using the Kaiser window method.

        .. ipython:: python

            fir = sdr.Interpolator(7); fir
            y = fir(x)

            @savefig sdr_Interpolator_1.png
            plt.figure(); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, marker=".", label="Interpolated"); \
            plt.title("Interpolation by 7 with the Kaiser window method");

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
            plt.figure(); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y1, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 0, marker=".", label="Interpolated $y_1[n]$"); \
            sdr.plot.time_domain(y2, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 10, marker=".", label="Interpolated $y_2[n]$"); \
            sdr.plot.time_domain(y3, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 20, marker=".", label="Interpolated $y_3[n]$"); \
            sdr.plot.time_domain(y4, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 30, marker=".", label="Interpolated $y_4[n]$"); \
            sdr.plot.time_domain(y5, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 40, marker=".", label="Interpolated $y_5[n]$"); \
            plt.title("Streaming interpolation by 7 with the Kaiser window method");

        Create a polyphase filter that interpolates by 7 using linear method.

        .. ipython:: python

            fir = sdr.Interpolator(7, "linear"); fir
            y = fir(x)

            @savefig sdr_Interpolator_3.png
            plt.figure(); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, marker=".", label="Interpolated"); \
            plt.title("Interpolation by 7 with the linear method");

        Create a polyphase filter that interpolates by 7 using the zero-order hold method. It is recommended to use
        the `"full"` convolution mode. This way the first upsampled symbol has $r$ samples.

        .. ipython:: python

            fir = sdr.Interpolator(7, "zoh"); fir
            y = fir(x, mode="full")

            @savefig sdr_Interpolator_4.png
            plt.figure(); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, offset=-fir.delay/fir.rate, marker=".", label="Interpolated"); \
            plt.title("Interpolation by 7 with the zero-order hold method");

    Group:
        dsp-polyphase-filtering
    """

    def __init__(
        self,
        interpolation: int,
        taps: Literal["kaiser", "linear", "linear-matlab", "zoh"] | npt.ArrayLike = "kaiser",
        polyphase_order: int = 23,
        atten: float = 80,
        streaming: bool = False,
    ):
        r"""
        Creates a polyphase FIR interpolating filter.

        Arguments:
            interpolation: The interpolation rate $P$.
            taps: The prototype filter design specification.

                - `"kaiser"`: The prototype filter is designed using :func:`sdr.multirate_fir`
                  with arguments `interpolation` and 1.
                - `"linear"`: The prototype filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2P$ linear ramp $\frac{1}{P} [0, ..., P-1, P, P-1, ..., 1]$.
                  The first output sample aligns with the first input sample.
                - `"linear-matlab"`: The prototype filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2P$ linear ramp $\frac{1}{P} [1, ..., P-1, P, P-1, ..., 0]$.
                  This is method MATLAB uses. The first output sample is advanced from the first input sample.
                - `"zoh"`: The prototype filter is designed to be a zero-order hold.
                  The filter coefficients are a length-$P$ array of ones.
                - `npt.ArrayLike`: The prototype filter feedforward coefficients $h[n]$.

            polyphase_order: The order of each polyphase filter. Must be odd, such that the filter lengths are even.
                Only used when `taps="kaiser"`.
            atten: The stopband attenuation $A_{\text{stop}}$ in dB. Only used when `taps="kaiser"`.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~Interpolator.__call__()`.
        """
        verify_scalar(interpolation, int=True, positive=True)
        verify_scalar(polyphase_order, int=True, positive=True)
        verify_scalar(atten, float=True, positive=True)
        verify_bool(streaming)

        self._method: Literal["kaiser", "linear", "linear-matlab", "zoh", "custom"]
        if not isinstance(taps, str):
            self._method = "custom"
            taps = verify_arraylike(taps, atleast_1d=True, ndim=1)
        else:
            self._method = verify_literal(taps, ["kaiser", "linear", "linear-matlab", "zoh"])
            if taps == "kaiser":
                taps = multirate_fir(interpolation, 1, polyphase_order, atten)
            elif taps == "linear":
                taps = multirate_fir_linear(interpolation)
            elif taps == "linear-matlab":
                taps = multirate_fir_linear_matlab(interpolation)
            elif taps == "zoh":
                taps = multirate_fir_zoh(interpolation)

        super().__init__(interpolation, taps, input="hold", output="top-to-bottom", streaming=streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    def __repr__(self) -> str:
        if self.method == "custom":
            h_str = np.array2string(self.taps, max_line_width=int(1e6), separator=", ", suppress_small=True)
        else:
            h_str = repr(self.method)
        return f"sdr.{type(self).__name__}({self.interpolation}, {h_str}, streaming={self.streaming})"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  rate: {self.interpolation} / {self.decimation}"
        string += f"\n  method: {self.method!r}"
        string += f"\n  polyphase_taps: {self.polyphase_taps.shape} shape"
        string += f"\n  delay: {self.delay}"
        string += f"\n  streaming: {self.streaming}"
        return string

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def method(self) -> Literal["kaiser", "linear", "linear-matlab", "zoh", "custom"]:
        """
        The method used to design the polyphase interpolating filter.
        """
        return self._method


@export
class Decimator(PolyphaseFIR):
    r"""
    Implements a polyphase decimating FIR filter.

    Notes:
        The polyphase decimating filter is equivalent to first filtering the input signal $x[n]$ with the
        prototype FIR filter with feedforward coefficients $h[n]$ and then downsampling the filtered signal
        by $Q$ (by discarding $Q-1$ samples every $Q$ samples).

        Instead, the polyphase decimating filter first decomposes the prototype FIR filter into $Q$ polyphase
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
            y[n] = Output signal with sample rate fs / Q
            h[n] = Prototype FIR filter
            @ = Adder

        The polyphase feedforward taps $h_i[n]$ are related to the prototype feedforward taps $h[n]$ by

        $$h_i[j] = h[i + j Q] .$$

    References:
        - fred harris, *Multirate Signal Processing for Communication Systems*, Chapter 7: Resampling Filters.

    Examples:
        Create an input signal to interpolate.

        .. ipython:: python

            x = np.cos(np.pi / 64 * np.arange(280))

        Create a polyphase filter that decimates by 7 using the Kaiser window method.

        .. ipython:: python

            fir = sdr.Decimator(7); fir
            y = fir(x)

            @savefig sdr_Decimator_1.png
            plt.figure(); \
            sdr.plot.time_domain(x, marker=".", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, marker="o", label="Decimated"); \
            plt.title("Decimation by 7 with the Kaiser window method");

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
            plt.figure(); \
            sdr.plot.time_domain(x, marker=".", label="Input"); \
            sdr.plot.time_domain(y1, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 0, marker="o", label="Decimated $y_1[n]$"); \
            sdr.plot.time_domain(y2, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 70, marker="o", label="Decimated $y_2[n]$"); \
            sdr.plot.time_domain(y3, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 140, marker="o", label="Decimated $y_3[n]$"); \
            sdr.plot.time_domain(y4, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 210, marker="o", label="Decimated $y_4[n]$"); \
            sdr.plot.time_domain(y5, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 280, marker="o", label="Decimated $y_5[n]$"); \
            plt.title("Streaming decimation by 7 with the Kaiser window method");

    Group:
        dsp-polyphase-filtering
    """

    def __init__(
        self,
        decimation: int,
        taps: Literal["kaiser"] | npt.ArrayLike = "kaiser",
        polyphase_order: int = 23,
        atten: float = 80,
        streaming: bool = False,
    ):
        r"""
        Creates a polyphase FIR decimating filter.

        Arguments:
            decimation: The decimation rate $Q$.
            taps: The prototype filter design specification.

                - `"kaiser"`: The prototype filter is designed using :func:`sdr.multirate_fir`
                  with arguments 1 and `decimation`.
                - `npt.ArrayLike`: The prototype filter feedforward coefficients $h[n]$.

            polyphase_order: The order of each polyphase filter. Must be odd, such that the filter lengths are even.
                Only used when `taps="kaiser"`.
            atten: The stopband attenuation $A_{\text{stop}}$ in dB. Only used when `taps="kaiser"`.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~Decimator.__call__()`.
        """
        verify_scalar(decimation, int=True, positive=True)

        self._method: Literal["kaiser", "custom"]
        if not isinstance(taps, str):
            self._method = "custom"
            taps = verify_arraylike(taps, atleast_1d=True, ndim=1)
        else:
            self._method = verify_literal(taps, ["kaiser"])
            taps = multirate_fir(1, decimation, polyphase_order, atten)

        super().__init__(decimation, taps, input="bottom-to-top", output="sum", streaming=streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    def __repr__(self) -> str:
        if self.method == "custom":
            h_str = np.array2string(self.taps, max_line_width=int(1e6), separator=", ", suppress_small=True)
        else:
            h_str = repr(self.method)
        return f"sdr.{type(self).__name__}({self.decimation}, {h_str}, streaming={self.streaming})"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  rate: {self.interpolation} / {self.decimation}"
        string += f"\n  method: {self.method!r}"
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
    def method(self) -> Literal["kaiser", "custom"]:
        """
        The method used to design the polyphase decimating filter.
        """
        return self._method


@export
class Resampler(PolyphaseFIR):
    r"""
    Implements a polyphase rational resampling FIR filter.

    Notes:
        The polyphase rational resampling filter is equivalent to first upsampling the input signal $x[n]$ by $P$
        (by inserting $P-1$ zeros between each sample), filtering the upsampled signal with the prototype FIR filter
        with feedforward coefficients $h[n]$, and then downsampling the filtered signal by $Q$ (by discarding $Q-1$
        samples every $Q$ samples).

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

    References:
        - fred harris, *Multirate Signal Processing for Communication Systems*, Chapter 7: Resampling Filters.

    Examples:
        Create an input signal to resample.

        .. ipython:: python

            x = np.cos(np.pi / 4 * np.arange(40))

        Create a polyphase filter that resamples by 7/3 using the Kaiser window method.

        .. ipython:: python

            fir = sdr.Resampler(7, 3); fir
            y = fir(x)

            @savefig sdr_Resampler_1.png
            plt.figure(); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, marker=".", label="Resampled"); \
            plt.title("Resampling by 7/3 with the Kaiser window method");

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
            plt.figure(); \
            sdr.plot.time_domain(x, marker="o", label="Input"); \
            sdr.plot.time_domain(y1, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 0, marker=".", label="Resampled $y_1[n]$"); \
            sdr.plot.time_domain(y2, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 10, marker=".", label="Resampled $y_2[n]$"); \
            sdr.plot.time_domain(y3, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 20, marker=".", label="Resampled $y_3[n]$"); \
            sdr.plot.time_domain(y4, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 30, marker=".", label="Resampled $y_4[n]$"); \
            sdr.plot.time_domain(y5, sample_rate=fir.rate, offset=-fir.delay/fir.rate + 40, marker=".", label="Resampled $y_5[n]$"); \
            plt.title("Streaming resampling by 7/3 with the Kaiser window method");

        Create a polyphase filter that resamples by 5/7 using linear method.

        .. ipython:: python

            fir = sdr.Resampler(5, 7); fir
            y = fir(x)

            @savefig sdr_Resampler_3.png
            plt.figure(); \
            sdr.plot.time_domain(x, marker=".", label="Input"); \
            sdr.plot.time_domain(y, sample_rate=fir.rate, marker="o", label="Resampled"); \
            plt.title("Resampling by 5/7 with the Kaiser window method");

    Group:
        dsp-polyphase-filtering
    """

    def __init__(
        self,
        interpolation: int,
        decimation: int,
        taps: Literal["kaiser", "linear", "linear-matlab", "zoh"] | npt.ArrayLike = "kaiser",
        polyphase_order: int = 23,
        atten: float = 80,
        streaming: bool = False,
    ):
        r"""
        Creates a polyphase FIR rational resampling filter.

        Arguments:
            interpolation: The interpolation rate $P$.
            decimation: The decimation rate $Q$.
            taps: The prototype filter design specification.

                - `"kaiser"`: The prototype filter is designed using :func:`sdr.multirate_fir`
                  with arguments `interpolation` and `decimation`.
                - `"linear"`: The prototype filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2P$ linear ramp $\frac{1}{P} [0, ..., P-1, P, P-1, ..., 1]$.
                  The first output sample aligns with the first input sample.
                - `"linear-matlab"`: The prototype filter is designed to linearly interpolate between samples.
                  The filter coefficients are a length-$2P$ linear ramp $\frac{1}{P} [1, ..., P-1, P, P-1, ..., 0]$.
                  This is method MATLAB uses. The first output sample is advanced from the first input sample.
                - `"zoh"`: The prototype filter is designed to be a zero-order hold.
                  The filter coefficients are a length-$P$ array of ones.
                - `npt.ArrayLike`: The prototype filter feedforward coefficients $h[n]$.

            polyphase_order: The order of each polyphase filter. Must be odd, such that the filter lengths are even.
                Only used when `taps="kaiser"`.
            atten: The stopband attenuation $A_{\text{stop}}$ in dB. Only used when `taps="kaiser"`.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~Resampler.__call__()`.
        """
        verify_scalar(interpolation, int=True, positive=True)
        verify_scalar(decimation, int=True, positive=True)

        self._method: Literal["kaiser", "linear", "linear-matlab", "zoh", "custom"]
        if not isinstance(taps, str):
            self._method = "custom"
            taps = verify_arraylike(taps, atleast_1d=True, ndim=1)
        else:
            self._method = verify_literal(taps, ["kaiser", "linear", "linear-matlab", "zoh"])
            if taps == "kaiser":
                taps = multirate_fir(interpolation, decimation, polyphase_order, atten)
            else:
                verify_scalar(interpolation, int=True, exclusive_min=1)
                if taps == "linear":
                    taps = multirate_fir_linear(interpolation)
                elif taps == "linear-matlab":
                    taps = multirate_fir_linear_matlab(interpolation)
                elif taps == "zoh":
                    taps = multirate_fir_zoh(interpolation)

        if interpolation == 1:
            # PolyphaseFIR configured like Decimator
            super().__init__(decimation, taps, input="bottom-to-top", output="sum", streaming=streaming)
        else:
            # PolyphaseFIR configured like Interpolator
            super().__init__(interpolation, taps, input="hold", output="top-to-bottom", streaming=streaming)
            self._decimation *= decimation  # Due to downsampling output of Interpolator

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike, mode: Literal["rate", "full"] = "rate") -> npt.NDArray:
        y = super().__call__(x, mode)
        if self.interpolation > 1:
            y = y[:: self.decimation]  # Downsample the interpolated output
        return y

    def __repr__(self) -> str:
        if self.method == "custom":
            h_str = np.array2string(self.taps, max_line_width=int(1e6), separator=", ", suppress_small=True)
        else:
            h_str = repr(self.method)
        return (
            f"sdr.{type(self).__name__}({self.interpolation}, {self.decimation}, {h_str}, streaming={self.streaming})"
        )

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  rate: {self.interpolation} / {self.decimation}"
        string += f"\n  method: {self.method!r}"
        string += f"\n  polyphase_taps: {self.polyphase_taps.shape} shape"
        string += f"\n  delay: {self.delay}"
        string += f"\n  streaming: {self.streaming}"
        return string

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def method(self) -> Literal["kaiser", "linear", "linear-matlab", "zoh", "custom"]:
        """
        The method used to design the polyphase resampling filter.
        """
        return self._method


@export
class Channelizer(PolyphaseFIR):
    r"""
    Implements a polyphase channelizer FIR filter.

    Notes:
        The polyphase channelizer efficiently splits the input signal $x[n]$ with sample rate $f_s$ into $C$
        equally spaced channels. Each channel has a bandwidth of $f_s / C$.

        The polyphase channelizer is equivalent to first mixing the input signal $x[n]$ with $C$ complex exponentials
        with frequencies $f_i = -i \cdot f_s / C$, filtering the mixed signals with the prototype FIR filter
        with feedforward coefficients $h[n]$, and then downsampling the filtered signals by $C$ (by discarding $C-1$
        samples every $C$ samples).

        Instead, the polyphase channelizer first decomposes the prototype FIR filter into $C$ polyphase filters with
        feedforward coefficients $h_i[n]$. The polyphase filters are then applied to the commutated input signal $x[n]$
        in parallel. The outputs of the polyphase filters are then inverse Discrete Fourier transformed (IDFT) to
        produce the $C$ channelized output signals $y_i[n]$.

        .. code-block:: text
           :caption: Polyphase Channelizer FIR Filter Block Diagram

                                                                  +------+
                                     +------------------------+   |      |
            ..., x[6], x[3], x[0] -->| h[0], h[3], h[6], h[9] |-->|      |--> ..., y[0,1], y[0,0]
                                     +------------------------+   |      |
                                     +------------------------+   |      |
            ..., x[5], x[2], 0    -->| h[1], h[4], h[7], 0    |-->| IDFT |--> ..., y[1,1], y[1,0]
                                     +------------------------+   |      |
                                     +------------------------+   |      |
            ..., x[4], x[1], 0    -->| h[2], h[5], h[8], 0    |-->|      |--> ..., y[2,1], y[2,0]
                                     +------------------------+   |      |
                                                                  +------+

            Input Commutator                                                 Parallel Outputs
            (bottom-to-top)

            x[n] = Input signal with sample rate fs
            y[i,n] = Channel i output signal with sample rate fs / C
            h[n] = Prototype FIR filter

        The polyphase feedforward taps $h_i[n]$ are related to the prototype feedforward taps $h[n]$ by

        $$h_i[j] = h[i + j C] .$$

    References:
        - fred harris, *Multirate Signal Processing for Communication Systems*, Chapter 6.1: Channelizer.

    Examples:
        Create a channelizer with 10 channels.

        .. ipython:: python

            C = 10
            channelizer = sdr.Channelizer(C); channelizer

        Create an input signal. Each channel has a tone with increasing frequency. The amplitude of each tone also
        increases by 2 dB for each channel.

        .. ipython:: python

            x = np.random.randn(10_000) + 1j * np.random.randn(10_000)
            for i in range(C):
                x += sdr.linear(10 + 2 * i) * np.exp(1j * 2 * np.pi * (i + 0.25 / C * i) / C * np.arange(10_000))

        Plot the input signal and overlay the channel boundaries. Note, Channel 5 is centered at $f = 0.5$. So, it
        wraps from positive to negative frequencies.

        .. ipython:: python

            plt.figure(); \
            sdr.plot.periodogram(x, fft=1024, color="k", label="Input $x[n]$");
            for i in range(C):
                f_start = (i - 0.5) / C
                f_stop = (i + 0.5) / C
                if f_start > 0.5:
                    f_start -= 1
                    f_stop -= 1
                plt.fill_betweenx([0, 80], f_start, f_stop, alpha=0.2, label=f"Channel {i}")
            @savefig sdr_Channelizer_1.png
            plt.xticks(np.arange(-0.5, 0.6, 0.1)); \
            plt.legend(); \
            plt.title("Input signals spread across 10 channels");

        Channelize the input signal with sample rate $f_s$ into 10 channels, each with sample rate $f_s / 10$.

        .. ipython:: python

            Y = channelizer(x)
            x.shape, Y.shape

        .. ipython:: python

            plt.figure();
            for i in range(C):
                sdr.plot.periodogram(Y[i, :], fft=1024, label=f"Channel {i}")
            @savefig sdr_Channelizer_2.png
            plt.xticks(np.arange(-0.5, 0.6, 0.1)); \
            plt.title("Output signals from 10 channels");

    Group:
        dsp-polyphase-filtering
    """

    def __init__(
        self,
        channels: int,
        taps: Literal["kaiser"] | npt.ArrayLike = "kaiser",
        polyphase_order: int = 23,
        atten: float = 80,
        streaming: bool = False,
    ):
        r"""
        Creates a polyphase FIR channelizing filter.

        Arguments:
            channels: The number of channels $C$.
            taps: The prototype filter design specification.

                - `"kaiser"`: The prototype filter is designed using :func:`sdr.multirate_fir`
                  with arguments 1 and `rate`.
                - `npt.ArrayLike`: The prototype filter feedforward coefficients $h[n]$.

            polyphase_order: The order of each polyphase filter. Must be odd, such that the filter lengths are even.
                Only used when `taps="kaiser"`.
            atten: The stopband attenuation $A_{\text{stop}}$ in dB. Only used when `taps="kaiser"`.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~Channelizer.__call__()`.
        """
        self._channels = verify_scalar(channels, int=True, positive=True)

        self._method: Literal["kaiser", "custom"]
        if not isinstance(taps, str):
            self._method = "custom"
            taps = verify_arraylike(taps, atleast_1d=True, ndim=1)
        else:
            self._method = verify_literal(taps, ["kaiser"])
            taps = multirate_fir(1, channels, polyphase_order, atten)

        super().__init__(channels, taps, input="bottom-to-top", output="all", streaming=streaming)

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(
        self,
        x: npt.ArrayLike,
        mode: Literal["rate", "full"] = "rate",
    ) -> npt.NDArray:
        r"""
        Channelizes the input signal $x[n]$ with the polyphase FIR filter.

        Arguments:
            x: The input signal $x[n]$ with sample rate $f_s$ and length $L$.
            mode: The non-streaming convolution mode.

                - `"rate"`: The output signals $y_i[n]$ have length $L / C$, proportional to the number of channels $C$.
                  Output sample 0 aligns with input sample 0.
                - `"full"`: The full convolution is performed. The output signals $y_i[n]$ have length $(L + N) / C$,
                  where $N$ is the order of the multirate filter. Output sample :obj:`~sdr.Channelizer.delay` aligns
                  with input sample 0.

                In streaming mode, the `"full"` convolution is performed. However, for each $L$ input samples
                only $L / C$ output samples are produced per call. A final call to :meth:`~Channelizer.flush()`
                is required to flush the filter state.

        Returns:
            A 2-D array of channelized signals $y_i[n]$ with sample rate $f_s / C$. The output length is dictated by
            the `mode` argument.
        """
        Y = super().__call__(x, mode)
        Y = np.fft.ifft(Y, axis=0)
        return Y

    def __repr__(self) -> str:
        if self.method == "custom":
            h_str = np.array2string(self.taps, max_line_width=int(1e6), separator=", ", suppress_small=True)
        else:
            h_str = repr(self.method)
        return f"sdr.{type(self).__name__}({self.channels}, {h_str}, streaming={self.streaming})"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  channels: {self.channels}"
        string += f"\n  rate: {self.interpolation} / {self.decimation}"
        string += f"\n  method: {self.method!r}"
        string += f"\n  polyphase_taps: {self.polyphase_taps.shape} shape"
        string += f"\n  delay: {self.delay}"
        string += f"\n  streaming: {self.streaming}"
        return string

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def channels(self) -> float:
        """
        The number of channels $C$.
        """
        return self._channels

    @property
    def method(self) -> Literal["kaiser", "custom"]:
        """
        The method used to design the polyphase channelizing filter.
        """
        return self._method
