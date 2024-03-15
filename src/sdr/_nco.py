"""
A module for numerically-controlled oscillators (NCO).
"""

from __future__ import annotations

from typing import Any, overload

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from ._helper import export


@export
class NCO:
    r"""
    Implements a numerically-controlled oscillator (NCO).

    Notes:
        .. code-block:: text
           :caption: Numerically-Controlled Oscillator Block Diagram

                          constant           constant
                         increment            offset    p[n]
                             |                  |         |
                    +----+   v   +------+       v         v   +-----------+
            f[n] -->| K0 |-->@-->| z^-1 |---+-->@-------->@-->| output(.) |--> y[n]
                    +----+   ^   +------+   |                 +-----------+
                             |              |
                             +--------------+

            f[n] = Input frequency signal (radians/sample)
            p[n] = Input phase signal (radians)
            y[n] = Output complex signal
            K0 = NCO gain
            increment = Constant phase accumulation (radians/sample)
            offset = Absolute phase offset (radians)
            output(.) = Phase-to-output mapping function, either: ., sin(.), cos(.), exp(j .)
            z^-1 = Unit delay
            @ = Adder

    Examples:
        Create an NCO with a constant phase increment of $2 \pi / 20$ radians/sample and a constant phase offset
        of $\pi$ radians.

        .. ipython:: python

            nco = sdr.NCO(increment=2 * np.pi / 20, offset=np.pi); \
            y = nco.step(100)

            @savefig sdr_NCO_1.png
            plt.figure(); \
            sdr.plot.time_domain(y, marker="."); \
            plt.title("Constant frequency NCO");

        Create an NCO with a constant phase increment of 0 radians/sample and a constant phase offset
        of 0 radians. Then step the NCO with a FSK frequency signal.

        .. ipython:: python

            nco = sdr.NCO(); \
            freq = np.array([1, 2, -3, 0, 2, -1]) * 2 * np.pi / 60; \
            freq = np.repeat(freq, 20); \
            y = nco(freq=freq)

            @savefig sdr_NCO_2.png
            plt.figure(); \
            sdr.plot.time_domain(y, marker="."); \
            plt.title("NCO implementing CP-FSK modulation");

        Create an NCO with a constant phase increment of $2 \pi / 57$ radians/sample and a constant phase offset
        of 0 radians. Then step the NCO with a BPSK phase signal.

        .. ipython:: python

            nco = sdr.NCO(increment=2 * np.pi / 57); \
            phase = np.array([0, 1, 0, 1, 0, 1]) * np.pi; \
            phase = np.repeat(phase, 20); \
            y = nco(phase=phase)

            @savefig sdr_NCO_3.png
            plt.figure(); \
            sdr.plot.time_domain(y, marker="."); \
            plt.title("NCO implementing BPSK modulation");

        See the :ref:`phase-locked-loop` example.

    Group:
        pll
    """

    def __init__(self, gain: float = 1.0, increment: float = 0.0, offset: float = 0.0):
        r"""
        Creates a numerically-controlled oscillator (NCO).

        Arguments:
            gain: The NCO gain $K_0$.
            increment: The constant accumulation $\omega$ of the NCO in radians/sample.
            offset: The absolute offset $\theta$ of the NCO in radians.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        self._K0 = gain
        self._increment = increment
        self._offset = offset
        self._z_prev: float  # Will be updated in reset()
        self.reset()

    def reset(self):
        r"""
        Resets the NCO.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        self._z_prev = 0.0

    @overload
    def __call__(
        self,
        freq: npt.NDArray[np.float64] | None = None,
        phase: npt.NDArray[np.float64] | None = None,
        output: Literal["phase", "sine", "cosine"] = "complex-exp",
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def __call__(
        self,
        freq: npt.NDArray[np.float64] | None = None,
        phase: npt.NDArray[np.float64] | None = None,
        output: Literal["complex-exp"] = "complex-exp",
    ) -> npt.NDArray[np.complex128]: ...

    def __call__(
        self,
        freq: Any = None,
        phase: Any = None,
        output: Any = "complex-exp",
    ) -> Any:
        """
        Steps the NCO with variable frequency and/or phase signals.

        Arguments:
            freq: The variable frequency signal $f[n]$ in radians/sample. This input signal varies the per-sample
                phase increment of the NCO. If `None`, the signal is all zeros.
            phase: The variable phase signal $p[n]$ in radians. This input signal varies the per-sample phase offset
                of the NCO. If `None`, the signal is all zeros.
            output: The format of the output signal $y[n]$. Options are the accumulated phase, sine, cosine, or
                complex exponential.

        Returns:
            The output signal $y[n]$.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        if freq is None and phase is None:
            raise ValueError("At least one of 'freq' and 'phase' must be provided.")
        elif freq is None:
            freq = np.zeros_like(phase)
        elif phase is None:
            phase = np.zeros_like(freq)
        assert freq is not None and phase is not None  # Needed for type checking

        if not freq.size == phase.size:
            raise ValueError(f"Arguments 'freq' and 'phase' must have the same size, not {freq.size} and {phase.size}.")

        # Scale the input by the NCO gain and add the constant accumulation to every sample
        z = freq * self.gain + self.increment
        z = np.atleast_1d(z)

        # Insert the previous output in front of the current input. Then run a cumulative sum over all samples.
        z = np.insert(z, 0, self._z_prev)
        z = np.cumsum(z)
        z, self._z_prev = z[0:-1], z[-1]

        # Add the absolute offset to every sample
        y = z + self.offset + phase

        if output == "phase":
            pass
        elif output == "complex-exp":
            y = np.exp(1j * y)
        elif output == "sine":
            y = np.sin(y)
        elif output == "cosine":
            y = np.cos(y)
        else:
            raise ValueError(
                f"Argument 'output' must be one of 'phase', 'sine', 'cosine', or 'complex-exp', not {output!r}."
            )

        if y.size == 1:
            y = y[0]

        return y

    def step(self, N: int) -> npt.NDArray[np.complex128]:
        """
        Steps the NCO forward by $N$ samples.

        Arguments:
            N: The number of samples $N$ to step the NCO forward.

        Returns:
            The output complex signal $y[n]$.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        x = np.zeros(N)
        y = self(x)
        return y

    @property
    def gain(self) -> float:
        """
        (Settable) The NCO gain $K_0$.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._K0

    @gain.setter
    def gain(self, value: float):
        self._K0 = value

    @property
    def increment(self) -> float:
        r"""
        (Settable) The constant phase accumulation $\omega$ of the NCO in radians/sample.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._increment

    @increment.setter
    def increment(self, value: float):
        self._increment = value

    @property
    def offset(self) -> float:
        r"""
        (Settable) The absolute phase offset $\theta$ of the NCO in radians.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._offset

    @offset.setter
    def offset(self, value: float):
        self._offset = value
