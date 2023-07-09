"""
A module for numerically-controlled oscillators (NCO).
"""
import numpy as np


class NCO:
    r"""
    Implements a numerically-controlled oscillator (NCO).

    Notes:
        .. code-block:: text
           :caption: Numerically-Controlled Oscillator Block Diagram

                         increment           offset
                             |                 |
                    +----+   v                 v
            x[n] -->| K0 |-->@--------------+--@--> y[n]
                    +----+   ^              |
                             |   +------+   |
                             +---| z^-1 |---+
                                 +------+

            x[n] = Input signal (units)
            y[n] = Output signal (units)
            K0 = NCO gain
            increment = Constant accumulation (units/sample)
            offset = Absolute offset (units)
            z^-1 = Unit delay
            @ = Adder

    Group:
        pll
    """

    def __init__(self, K0: float = 1.0, increment: float = 0.0, offset: float = 0.0):
        """
        Creates a numerically-controlled oscillator (NCO).

        Arguments:
            K0: The NCO gain.
            increment: The constant accumulation of the NCO in units/sample.
            offset: The absolute offset of the NCO in units.
        """
        self._K0 = K0
        self._increment = increment
        self._offset = offset
        self._y_prev: float  # Will be updated in reset()
        self.reset()

    def reset(self):
        """
        Resets the NCO.
        """
        self._y_prev = 0.0

    def step(self, N: int) -> np.ndarray:
        """
        Steps the NCO forward by $N$ samples.

        Arguments:
            N: The number of samples to step the NCO forward.

        Returns:
            The output signal, $y[n]$.
        """
        x = np.zeros(N)
        y = self.process(x)
        return y

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Steps the NCO with the variable-increment signal $x[n]$.

        Arguments:
            x: The input signal, $x[n]$. The input signal varies the per-sample increment of the NCO.

        Returns:
            The output signal, $y[n]$.
        """
        x = np.atleast_1d(x)

        # Scale the input by the NCO gain and add the constant accumulation to every sample
        z = x * self.K0 + self.increment

        # Increment the first sample by the previous output. Then run a cumulative sum over all samples.
        z[0] += self._y_prev
        y = np.cumsum(z)

        # Add the absolute offset to every sample
        y += self.offset

        # Store the last output sample for the next iteration
        self._y_prev = y[-1]

        return y

    @property
    def K0(self) -> float:
        """
        The NCO gain.
        """
        return self._K0

    @K0.setter
    def K0(self, value: float):
        self._K0 = value

    @property
    def increment(self) -> float:
        """
        The constant accumulation of the NCO in units/sample.
        """
        return self._increment

    @increment.setter
    def increment(self, value: float):
        self._increment = value

    @property
    def offset(self) -> float:
        """
        The absolute offset of the NCO in units.
        """
        return self._offset

    @offset.setter
    def offset(self, value: float):
        self._offset = value
