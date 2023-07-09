"""
A module for numerically-controlled oscillators (NCO).
"""
import numpy as np

from ._helper import export


@export
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

    See Also:
        sdr.DDS

    Examples:
        See the :ref:`phase-locked-loop` example.

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

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        self._K0 = K0
        self._increment = increment
        self._offset = offset
        self._y_prev: float  # Will be updated in reset()
        self.reset()

    def reset(self):
        """
        Resets the NCO.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        self._y_prev = 0.0

    def step(self, N: int) -> np.ndarray:
        """
        Steps the NCO forward by $N$ samples.

        Arguments:
            N: The number of samples to step the NCO forward.

        Returns:
            The output signal, $y[n]$.

        Examples:
            See the :ref:`phase-locked-loop` example.
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

        Examples:
            See the :ref:`phase-locked-loop` example.
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

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._K0

    @K0.setter
    def K0(self, value: float):
        self._K0 = value

    @property
    def increment(self) -> float:
        """
        The constant accumulation of the NCO in units/sample.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._increment

    @increment.setter
    def increment(self, value: float):
        self._increment = value

    @property
    def offset(self) -> float:
        """
        The absolute offset of the NCO in units.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._offset

    @offset.setter
    def offset(self, value: float):
        self._offset = value


@export
class DDS:
    r"""
    Implements a direct digital synthesizer (DDS).

    Notes:
        .. code-block:: text
           :caption: Direct Digital Synthesizer Block Diagram

                         increment           offset
                             |                 |
                    +----+   v                 v   +--------+
            x[n] -->| K0 |-->@--------------+--@-->| e^(j.) |--> y[n]
                    +----+   ^              |      +--------+
                             |   +------+   |
                             +---| z^-1 |---+
                                 +------+

            x[n] = Input signal (radians)
            y[n] = Output signal (radians)
            K0 = NCO gain
            increment = Constant accumulation (radians/sample)
            offset = Absolute offset (radians)
            z^-1 = Unit delay
            @ = Adder

    See Also:
        sdr.NCO

    Examples:
        See the :ref:`phase-locked-loop` example.

    Group:
        pll
    """

    def __init__(self, K0: float = 1.0, increment: float = 0.0, offset: float = 0.0):
        """
        Creates a direct digital synthesizer (DDS).

        Arguments:
            K0: The NCO gain.
            increment: The constant accumulation of the NCO in radians/sample.
            offset: The absolute offset of the NCO in radians.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        self._nco = NCO(K0, increment, offset)
        self.reset()

    def reset(self):
        """
        Resets the DDS.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        self.nco.reset()

    def step(self, N: int) -> np.ndarray:
        """
        Steps the DDS forward by $N$ samples.

        Arguments:
            N: The number of samples to step the DDS forward.

        Returns:
            The output complex exponential, $y[n]$.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        x = np.zeros(N)
        y = self.process(x)
        return y

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Steps the DDS with the variable phase increment signal $x[n]$.

        Arguments:
            x: The input signal, $x[n]$. The input signal varies the per-sample phase increment of the DDS.

        Returns:
            The output complex exponential, $y[n]$.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        phase = self.nco.process(x)
        y = np.exp(1j * phase)
        return y

    @property
    def nco(self) -> NCO:
        """
        The numerically-controlled oscillator (NCO) used by the DDS.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._nco
