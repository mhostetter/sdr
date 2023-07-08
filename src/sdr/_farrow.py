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

    Examples:
        See the :ref:`farrow-arbitrary-resampler` example.

    Group:
        resampling
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

        Arguments:
            state: The filter state to reset to. The state vector should equal the previous three
                inputs. If `None`, the filter state will be reset to zero.
        """
        if state is None:
            self._x_prev = np.zeros(self._taps.shape[1] - 1, dtype=np.float32)
        else:
            state = np.asarray(state, dtype=np.float32)
            if not state.size == self._taps.shape[1] - 1:
                raise ValueError(f"Argument 'state' must have {self._taps.shape[1]} elements, not {state.size}.")
            self._x_prev = state

        # Initial fractional sample delay accounts for filter delay
        self._mu_next = self._taps.shape[1] // 2

    def resample(self, x: np.ndarray, rate: float) -> np.ndarray:
        r"""
        Resamples the input signal by the given arbitrary rate.

        Arguments:
            x: The input signal, $x[n] = x(n T_s)$.
            rate: The resampling rate, $r$.

        Returns:
            The resampled signal, $y[n] = x(n \frac{T_s}{r})$.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
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
            # Compute the four FIR filter outputs for the entire input signal
            y0 = scipy.signal.convolve(x, self._taps[0, :], mode="full")
            y1 = scipy.signal.convolve(x, self._taps[1, :], mode="full")
            y2 = scipy.signal.convolve(x, self._taps[2, :], mode="full")
            y3 = scipy.signal.convolve(x, self._taps[3, :], mode="full")

            # Compute the fractional sample indices for each output sample
            mu = np.arange(
                self._taps.shape[1] // 2,  # Account for filter delay
                self._taps.shape[1] // 2 + x.size,
                1 / rate,
            )

        # Convert the fractional indices to integer indices and fractional indices
        idxs = (mu // 1.0).astype(int)
        mu = mu - idxs
        mu *= -1  # TODO: Why is this the case?

        # Interpolate the output samples using the Horner method
        y = mu * (mu * (mu * y0[idxs] + y1[idxs]) + y2[idxs]) + y3[idxs]

        return y

    @property
    def streaming(self) -> bool:
        """
        Returns whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :obj:`resample()`.

        Examples:
            .. ipython:: python

                farrow = sdr.FarrowResampler()
                farrow.streaming
        """
        return self._streaming

    @property
    def order(self) -> int:
        """
        Returns the order of the filter.

        Examples:
            .. ipython:: python

                farrow = sdr.FarrowResampler()
                farrow.order
        """
        return self._taps.shape[1] - 1

    @property
    def taps(self) -> np.ndarray:
        """
        Returns the Farrow filter taps.

        Examples:
            .. ipython:: python

                farrow = sdr.FarrowResampler()
                farrow.taps
        """
        return self._taps
