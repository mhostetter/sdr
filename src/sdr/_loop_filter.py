"""
A module for discrete-time loop filters.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._filter import IIR
from ._helper import export


@export
class LoopFilter:
    r"""
    Implements a 2nd order, proportional-plus-integrator (PPI) loop filter.

    Notes:
        .. code-block:: text
            :caption: Proportional-Plus-Integral Loop Filter Block Diagram

                       +----+
                   +-->| K1 |-------------------+
                   |   +----+                   |
            x[n] --+                            @--> y[n]
                   |   +----+                   |
                   +-->| K2 |--@-------------+--+
                       +----+  ^             |
                               |  +------+   |
                               +--| z^-1 |<--+
                                  +------+

            x[n] = Input signal
            y[n] = Output signal
            K1 = Proportional gain
            K2 = Integral gain
            z^-1 = Unit delay
            @ = Adder

        The transfer function of the loop filter is

        $$H(z) = K_1 + K_2 \frac{ 1 }{ 1 - z^{-1}} = \frac{(K_1 + K_2) - K_1 z^{-1}}{1 - z^{-1}} .$$

        The second-order proportional-plus-integrator loop filter can track a constant phase error
        and/or frequency error to zero. It cannot, however, track a constant chirp (frequency ramp)
        to zero.

    References:
        - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Appendix C: Phase Locked Loops.

    Examples:
        See the :ref:`phase-locked-loop` example.

    Group:
        pll
    """

    def __init__(self, noise_bandwidth: float, damping_factor: float, K0: float = 1.0, Kp: float = 1.0):
        r"""
        Creates a 2nd order, proportional-plus-integrator (PPI) loop filter.

        Arguments:
            noise_bandwidth: The normalized noise bandwidth $B_n T$ of the loop filter,
                where $B_n$ is the noise bandwidth in Hz and $T$ is the sampling period in seconds.
            damping_factor: The damping factor $\zeta$ of the loop filter. $\zeta = 1$ is critically damped,
                $\zeta < 1$ is underdamped, and $\zeta > 1$ is overdamped.
            K0: The NCO gain $K_0$.
            Kp: The gain $K_p$ of the phase error detector (PED) or time error detector (TED).

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        BnT = noise_bandwidth
        zeta = damping_factor

        # Equation C.57, page 736
        theta_n = BnT / (zeta + 1 / (4 * zeta))

        # Equation C.58, page 737
        K1 = 4 * zeta * theta_n / (1 + 2 * zeta * theta_n + theta_n**2) / K0 / Kp
        K2 = 4 * theta_n**2 / (1 + 2 * zeta * theta_n + theta_n**2) / K0 / Kp

        b = [K1 + K2, -K1]
        a = [1, -1]

        self._BnT = BnT
        self._zeta = zeta
        self._K1 = K1
        self._K2 = K2
        self._iir = IIR(b, a, streaming=True)

        self.reset()

    def reset(self) -> None:
        """
        Resets the loop filter.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        self.iir.reset()

    def __call__(self, x: npt.ArrayLike) -> np.ndarray:
        """
        Filters the input signal $x[n]$.

        Arguments:
            x: The input signal $x[n]$.

        Returns:
            The filtered output signal $y[n]$.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self.iir(x)

    @property
    def noise_bandwidth(self) -> float:
        """
        The normalized noise bandwidth $B_n T$ of the loop filter,
        where $B_n$ is the noise bandwidth in Hz and $T$ is the sampling period in seconds.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._BnT

    @property
    def damping_factor(self) -> float:
        r"""
        The damping factor $\zeta$ of the loop filter. $\zeta = 1$ is critically damped,
        $\zeta < 1$ is underdamped, and $\zeta > 1$ is overdamped.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._zeta

    @property
    def K1(self) -> float:
        """
        The proportional gain $K_1$ of the loop filter.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._K1

    @property
    def K2(self) -> float:
        """
        The integral gain $K_2$ of the loop filter.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._K2

    @property
    def iir(self) -> IIR:
        """
        The underlying IIR filter used to implement the loop filter.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._iir
