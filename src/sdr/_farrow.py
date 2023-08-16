"""
A module containing a Farrow arbitrary resampler.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal

from ._helper import export


@export
class FarrowResampler:
    """
    Implements a cubic Farrow arbitrary resampler.

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

        Arguments:
            streaming: Indicates whether to use streaming mode. In streaming mode, previous inputs are
                preserved between calls to :meth:`~FarrowResampler.__call__()`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
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
            dtype=float,
        )

        self.reset()

    def reset(self, state: npt.ArrayLike | None = None):
        """
        *Streaming-mode only:* Resets the filter state and fractional sample index.

        Arguments:
            state: The filter state to reset to. The state vector should equal the previous three
                inputs. If `None`, the filter state will be reset to zero.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        if state is None:
            self._x_prev = np.zeros(self._taps.shape[1] - 1)
        else:
            state = np.asarray(state)
            if not state.size == self._taps.shape[1] - 1:
                raise ValueError(f"Argument 'state' must have {self._taps.shape[1]} elements, not {state.size}.")
            self._x_prev = state

        # Initial fractional sample delay accounts for filter delay
        self._mu_next = self._taps.shape[1] // 2

    def __call__(self, x: npt.ArrayLike, rate: float) -> np.ndarray:
        r"""
        Resamples the input signal $x[n]$ by the given arbitrary rate $r$.

        Arguments:
            x: The input signal $x[n] = x(n T_s)$.
            rate: The resampling rate $r$.

        Returns:
            The resampled signal $y[n] = x(n T_s / r)$.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        if not isinstance(rate, (int, float)):
            raise TypeError("Argument 'rate' must be an integer or float.")
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

            # Compute the fractional sample indices for each output sample. We want to step from mu_next to
            # y0.size in steps of 1 / rate. We configure the arange() such that the step is at least 1.
            if rate > 1:
                mu = np.arange(self._mu_next * rate, y0.size * rate) / rate
            else:
                mu = np.arange(self._mu_next, y0.size, 1 / rate)

            # Store the previous inputs and next fractional sample index for the next call to __call__()
            self._x_prev = x_pad[-(self._taps.shape[1] - 1) :]
            self._mu_next = (mu[-1] + 1 / rate) - y0.size
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
    def taps(self) -> np.ndarray:
        """
        The Farrow filter taps.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._taps

    @property
    def order(self) -> int:
        """
        The order of the filter.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._taps.shape[1] - 1

    @property
    def streaming(self) -> bool:
        """
        Indicates whether the filter is in streaming mode.

        In streaming mode, the filter state is preserved between calls to :meth:`~FarrowResampler.__call__()`.

        Examples:
            See the :ref:`farrow-arbitrary-resampler` example.
        """
        return self._streaming
