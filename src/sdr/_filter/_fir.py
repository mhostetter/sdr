"""
A module for finite impulse response (FIR) filters.
"""

from __future__ import annotations

from typing import Any, overload

import numpy as np
import numpy.typing as npt
import scipy.signal
from typing_extensions import Literal

from .._helper import export
from ._common import frequency_response


@export
class FIR:
    r"""
    Implements a finite impulse response (FIR) filter.

    This class is a wrapper for the :func:`scipy.signal.convolve` function. It supports one-time filtering
    and streamed filtering.

    Notes:
        A FIR filter is defined by its feedforward coefficients $h[n]$.

        $$y[n] = \sum_{i=0}^{N} h[i] \cdot x[n-i] .$$

        The transfer function of the filter is

        $$H(z) = \sum\limits_{i=0}^{N} h[i] \cdot z^{-i} .$$

        .. code-block:: text
            :caption: FIR Block Diagram

                      +------+    +------+            +------+    +------+
            x[n] --+->| z^-1 |-+->| z^-1 |-+--...--+->| z^-1 |-+->| z^-1 |-+
                   |  +------+ |  +------+ |       |  +------+ |  +------+ |
                   |           |           |       |           |           |
              h[0] |      h[1] |      h[2] |       |    h[N-1] |      h[N] |
                   |           |           |       |           |           |
                   +---------->@---------->@--...->@---------->@---------->@---> y[n]

    Examples:
        See the :ref:`fir-filters` example.

    Group:
        dsp-fir-filtering
    """

    def __init__(self, h: npt.ArrayLike, streaming: bool = False):
        """
        Creates an FIR filter.

        Arguments:
            h: The feedforward coefficients $h[n]$.
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~FIR.__call__()`.

        Examples:
            See the :ref:`fir-filters` example.
        """
        self._taps = np.asarray(h)
        self._streaming = streaming
        self._delay = self.taps.size // 2

        self._state: npt.NDArray  # The filter state. Will be updated in reset().
        self.reset()

    ##############################################################################
    # Special methods
    ##############################################################################

    def __call__(self, x: npt.ArrayLike, mode: Literal["full", "valid", "same"] = "full") -> npt.NDArray:
        r"""
        Filters the input signal $x[n]$ with the FIR filter.

        Arguments:
            x: The input signal $x[n]$ with length $L$.
            mode: The non-streaming convolution mode.

                - `"same"`: The output signal $y[n]$ has length $L$. Output sample 0 aligns with input sample 0.
                - `"full"`: The full convolution is performed. The output signal $y[n]$ has length $L + N$,
                  where $N$ is the order of the filter. Output sample :obj:`~sdr.FIR.delay` aligns
                  with input sample 0.

                In streaming mode, the `"full"` convolution is performed. However, for each $L$ input samples
                only $L$ output samples are produced per call. A final call to :meth:`~sdr.FIR.flush()`
                is required to flush the filter state.

        Returns:
            The filtered signal $y[n]$. The output length is dictated by the `mode` argument.

        Examples:
            See the :ref:`fir-filters` example.
        """
        x = np.atleast_1d(x)
        if not x.ndim == 1:
            raise ValueError(f"Argument 'x' must be a 1-D, not {x.ndim}-D.")

        if self.streaming:
            # Prepend previous inputs from last __call__() call
            x_pad = np.concatenate((self._state, x))
            y = scipy.signal.convolve(x_pad, self.taps, mode="valid")
            self._state = x_pad[-(self.taps.size - 1) :]
        else:
            y = scipy.signal.convolve(x, self.taps, mode=mode)

        return y

    def __repr__(self) -> str:
        return f"sdr.{type(self).__name__}({self.taps.tolist()}, streaming={self.streaming})"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  taps: {self.taps.shape} shape"
        string += f"\n    {self.taps.tolist()}"
        string += f"\n  delay: {self.delay}"
        string += f"\n  streaming: {self.streaming}"
        return string

    def __len__(self) -> int:
        """
        Returns the filter length $N + 1$.

        Examples:
            See the :ref:`fir-filters` example.
        """
        return self.taps.size

    ##############################################################################
    # Streaming mode
    ##############################################################################

    def reset(self):
        """
        Resets the filter state. Only useful when using streaming mode.

        Examples:
            See the :ref:`fir-filters` example.

        Group:
            Streaming mode only
        """
        self._state = np.zeros(self.taps.size - 1, dtype=self.taps.dtype)

    def flush(self) -> npt.NDArray:
        """
        Flushes the filter state by passing zeros through the filter. Only useful when using streaming mode.

        Returns:
            The remaining filtered signal $y[n]$.

        Examples:
            See the :ref:`fir-filters` example.

        Group:
            Streaming mode only
        """
        x = np.zeros_like(self.state)
        y = self(x)
        return y

    @property
    def streaming(self) -> bool:
        """
        Indicates whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :meth:`~FIR.__call__()`.

        Examples:
            See the :ref:`fir-filters` example.

        Group:
            Streaming mode only
        """
        return self._streaming

    @property
    def state(self) -> npt.NDArray:
        """
        The filter state consisting of the previous $N$ inputs.

        Examples:
            See the :ref:`fir-filters` example.

        Group:
            Streaming mode only
        """
        return self._state

    ##############################################################################
    # Methods
    ##############################################################################

    def impulse_response(self, N: int | None = None) -> npt.NDArray:
        r"""
        Returns the impulse response $h[n]$ of the FIR filter.

        The impulse response $h[n]$ is the filter output when the input is an impulse $\delta[n]$.

        Arguments:
            N: The number of samples to return. The default is the filter length.

        Returns:
            The impulse response of the IIR filter $h[n]$.

        See Also:
            sdr.plot.impulse_response

        Examples:
            See the :ref:`fir-filters` example.
        """
        if N is None:
            N = self.taps.size

        if not N >= self.taps.size:
            raise ValueError("Argument 'N' must be greater than or equal to the filter length.")

        # Delta impulse function
        d = np.zeros(N - self.taps.size + 1, dtype=float)
        d[0] = 1

        h = scipy.signal.convolve(d, self.taps, mode="full")

        return h

    def step_response(self, N: int | None = None) -> npt.NDArray:
        """
        Returns the step response $s[n]$ of the FIR filter.

        The step response $s[n]$ is the filter output when the input is a unit step $u[n]$.

        Arguments:
            N: The number of samples to return. The default is the filter length.

        Returns:
            The step response of the FIR filter $s[n]$.

        See Also:
            sdr.plot.step_response

        Examples:
            See the :ref:`fir-filters` example.
        """
        if N is None:
            N = self.taps.size

        if not N >= self.taps.size:
            raise ValueError("Argument 'N' must be greater than or equal to the filter length.")

        # Unit step function
        u = np.ones(N - self.taps.size + 1, dtype=float)

        s = scipy.signal.convolve(u, self.taps, mode="full")

        return s

    @overload
    def frequency_response(
        self,
        freqs: int = 1024,
        sample_rate: float = 1.0,
        whole: bool = True,
        decades: int | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]: ...

    @overload
    def frequency_response(
        self,
        freqs: float,
        sample_rate: float = 1.0,
    ) -> complex: ...

    @overload
    def frequency_response(
        self,
        freqs: npt.NDArray[np.float64],
        sample_rate: float = 1.0,
    ) -> npt.NDArray[np.complex128]: ...

    def frequency_response(
        self,
        freqs: Any = 1024,
        sample_rate: Any = 1.0,
        whole: Any = True,
        decades: Any | None = None,
    ) -> Any:
        r"""
        Returns the frequency response $H(\omega)$ of the FIR filter.

        Arguments:
            freqs: The frequency specification.

                - `int`: The number of frequency points. The endpoint is not included.
                - `float`: A single frequency.
                - `npt.NDArray[float]`: Multiple frequencies.

            sample_rate: The sample rate $f_s$ of the filter in samples/s.
            whole: Only used if `freqs` is an integer.

                - `True`: The maximum frequency is `max_f = sample_rate`.
                - `False`: The maximum frequency is `max_f = sample_rate / 2`.

            decades: Only used if `freqs` is an integer.

                - `None`: `f = np.linspace(0, max_f, freqs, endpoint=False)`.
                - `int`: `f = np.logspace(np.log10(max_f) - decades), np.log10(max_f), freqs, endpoint=False)`.

        Returns:
            - The frequency vector $f$, only if `freqs` is an integer.
            - The frequency response of the FIR filter $H(\omega)$.

        See Also:
            sdr.plot.magnitude_response, sdr.plot.phase_response

        Examples:
            .. ipython:: python

                h = sdr.design_lowpass_fir(100, 0.2, window="hamming"); \
                fir = sdr.FIR(h)

            Compute the frequency response at 1024 evenly spaced frequencies.

            .. ipython:: python

                fir.frequency_response()

            Compute the frequency response at 0.0 rad/s.

            .. ipython:: python

                fir.frequency_response(0.0)

            Compute the frequency response at several frequencies in Hz.

            .. ipython:: python

                fir.frequency_response([100, 200, 300, 400], sample_rate=1000)
        """
        f, H = frequency_response(self.taps, 1, freqs, sample_rate, whole, decades)
        if isinstance(freqs, int):
            return f, H
        elif isinstance(freqs, float):
            return H[0]
        else:
            return H

    def group_delay(self, sample_rate: float = 1.0, N: int = 1024) -> tuple[npt.NDArray, npt.NDArray]:
        r"""
        Returns the group delay $\tau_g(\omega)$ of the FIR filter.

        Arguments:
            sample_rate: The sample rate $f_s$ of the filter in samples/s.
            N: The number of samples in the group delay.

        Returns:
            - The frequencies $f$ from $-f_s/2$ to $f_s/2$ in Hz.
            - The group delay of the FIR filter $\tau_g(\omega)$.

        See Also:
            sdr.plot.group_delay

        Examples:
            See the :ref:`fir-filters` example.
        """
        f, gd = scipy.signal.group_delay((self.taps, 1), w=N, whole=True, fs=sample_rate)

        f[f >= 0.5 * sample_rate] -= sample_rate  # Wrap frequencies from [0, 1) to [-0.5, 0.5)
        f = np.fft.fftshift(f)
        gd = np.fft.fftshift(gd)

        return f, gd

    def phase_delay(self, sample_rate: float = 1.0, N: int = 1024) -> tuple[npt.NDArray, npt.NDArray]:
        r"""
        Returns the phase delay $\tau_{\phi}(\omega)$ of the FIR filter.

        Arguments:
            sample_rate: The sample rate $f_s$ of the filter in samples/s.
            N: The number of samples in the phase delay.

        Returns:
            - The frequencies $f$ from $-f_s/2$ to $f_s/2$ in Hz.
            - The phase delay of the FIR filter $\tau_{\phi}(\omega)$.

        See Also:
            sdr.plot.phase_delay

        Examples:
            See the :ref:`fir-filters` example.
        """
        f, H = scipy.signal.freqz(self.taps, 1, worN=N, whole=True, fs=sample_rate)

        f -= sample_rate / 2
        H = np.fft.fftshift(H)

        theta = np.unwrap(np.angle(H), period=np.pi)
        theta -= theta[np.argmin(np.abs(f))]  # Set omega=0 to have phase of 0
        tau_phi = -theta / (2 * np.pi * f)
        tau_phi[np.argmin(np.abs(f))] = np.nan  # Avoid crazy result when dividing by near zero

        return f, tau_phi

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def taps(self) -> npt.NDArray:
        """
        The feedforward taps $h[n]$ with length $N + 1$.

        Examples:
            See the :ref:`fir-filters` example.
        """
        return self._taps

    @property
    def order(self) -> int:
        """
        The order of the FIR filter $N$.

        Examples:
            See the :ref:`fir-filters` example.
        """
        return self.taps.size - 1

    @property
    def delay(self) -> int:
        r"""
        The delay of the FIR filter $d = \lfloor \frac{N + 1}{2} \rfloor$ in samples.

        Examples:
            See the :ref:`fir-filters` example.
        """
        return self._delay
