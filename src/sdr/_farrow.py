"""
A module containing a Farrow arbitrary resampler.
"""
from __future__ import annotations

import numpy as np
import scipy.signal


class FarrowResampler:
    """
    A cubic Farrow arbitrary resampler.

    References:
        - https://wirelesspi.com/fractional-delay-filters-using-the-farrow-structure/
    """

    def __init__(self, streaming: bool = False):
        """
        Creates a new Farrow arbitrary resampler.
        """
        self._streaming = streaming
        self._x_prev: np.ndarray  # FIR filter state. Will be updated in reset().
        self._mu_next: float  # The next fractional sample delay value

        # Construct the four FIR filter taps
        self._taps = np.array(
            [
                [-1 / 6, 1 / 2, -1 / 2, 1 / 6],
                [0, 1 / 2, -1, 1 / 2],
                [1 / 6, -1, 1 / 2, 1 / 3],
                [0, 0, 1, 0],
            ],
            dtype=np.float32,
        )

        self.reset()

    def reset(self, state: np.ndarray | None = None):
        """
        Resets the filter state and fractional sample index. Only useful for streaming mode.
        """
        if state is None:
            self._x_prev = np.zeros(4, dtype=np.float32)
        else:
            state = np.asarray(state, dtype=np.float32)
            if not state.size == self._taps.shape[1]:
                raise ValueError(f"Argument 'state' must have {self._taps.shape[1]} elements, not {state.size}.")
            self._x_prev = state

        self._mu_next = 0.0

    def resample(self, x: np.ndarray, rate: float, mode="full") -> np.ndarray:
        """
        Resamples the input signal by the given arbitrary rate.

        Arguments:
            x: The input signal.
            rate: The resampling rate.
        """
        if not rate > 0:
            raise ValueError("Argument 'rate' must be positive.")

        x = np.atleast_1d(x)
        if self.streaming:
            # Prepend previous inputs from last streaming call
            x_pad = np.concatenate((self._x_prev, x))

            # Compute the four FIR filter outputs, but only keep the valid outputs
            y0 = scipy.signal.convolve(x_pad, self._taps[0, :], mode="valid")
            y1 = scipy.signal.convolve(x_pad, self._taps[1, :], mode="valid")
            y2 = scipy.signal.convolve(x_pad, self._taps[2, :], mode="valid")
            y3 = scipy.signal.convolve(x_pad, self._taps[3, :], mode="valid")
            self._x_prev = x_pad[-(self._taps.shape[1] - 1) :]

            # Compute the fractional sample indices for each output sample
            mu = np.arange(self._mu_next, x.size, 1 / rate)

            # Store the next fractional sample index for next call to resample()
            self._mu_next = (mu[-1] + 1 / rate) % 1.0
        else:
            # Compute the four FIR filter outputs for the entire input signal, using the specified
            # convolution mode
            y0 = scipy.signal.convolve(x, self._taps[0, :], mode=mode)
            y1 = scipy.signal.convolve(x, self._taps[1, :], mode=mode)
            y2 = scipy.signal.convolve(x, self._taps[2, :], mode=mode)
            y3 = scipy.signal.convolve(x, self._taps[3, :], mode=mode)

            # Compute the fractional sample indices for each output sample
            mu = np.arange(0, x.size + self._taps.shape[1] - 1, 1 / rate)

        # Convert the fractional indices to integer indices and fractional indices
        idx = (mu // 1.0).astype(int)
        mu = mu - idx
        mu *= -1  # TODO: Why is this the case?

        # Interpolate the output samples using the Horner method
        y = mu * (mu * (mu * y0[idx] + y1[idx]) + y2[idx]) + y3[idx]

        return y

    @property
    def streaming(self) -> bool:
        """
        Returns whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :obj:`resample()`.

        .. ipython:: python

            farrow = sdr.FarrowResampler()
            farrow.streaming
        """
        return self._streaming

    @property
    def order(self) -> int:
        """
        Returns the order of the filter.

        .. ipython:: python

            farrow = sdr.FarrowResampler()
            farrow.order
        """
        return self._taps.shape[1] - 1

    @property
    def taps(self) -> np.ndarray:
        """
        Returns the Farrow filter taps.

        .. ipython:: python

            farrow = sdr.FarrowResampler()
            farrow.taps
        """
        return self._taps
