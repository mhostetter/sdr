"""
A module for closed-loop phase-locked loops (PLLs) analysis.
"""
from __future__ import annotations

import numpy as np

from ._conversion import linear
from ._filter import IIR
from ._helper import export
from ._loop_filter import LoopFilter


@export
class ClosedLoopPLL:
    r"""
    A class that defines the performance of a closed-loop PLL.

    Note:
        This class is meant for performance analysis only.

    Notes:
        .. code-block:: text
            :caption: Closed-Loop PLL Block Diagram

                         bb[n]
                    +---+    +-----+    +----+
            x[n] -->| X |--->| PED |--->| LF |---+
                    +---+    +-----+    +----+   |
                      ^                          |
                      |  +---------+   +-----+   |
               lo[n]  +--| e^(-j.) |<--| NCO |<--+
                         +---------+   +-----+

            x[n] = Input signal
            lo[n] = Local oscillator signal
            bb[n] = Baseband signal
            PED = Phase error detector
            LF = Loop filter
            NCO = Numerically-controlled oscillator

        The transfer function of the 2nd order, proportional-plus-integrator loop filter is

        $$H_{LF}(z) = K_1 + K_2 \frac{ 1 }{ 1 - z^{-1}} = \frac{(K_1 + K_2) - K_1 z^{-1}}{1 - z^{-1}} .$$

        The transfer function of the NCO is

        $$H_{NCO}(z) = K_0 \frac{z^{-1}}{1 - z^{-1}} .$$

        The closed-loop transfer function of the PLL is

        $$
        H_{PLL}(z) = \frac{K_p K_0 (K_1 + K_2) z^{-1} - K_p K_0 K_1 z^{-2}}
        {1 - 2 (1 - \frac{1}{2} K_p K_0 (K_1 + K_2) z^{-1} + (1 - K_p K_0 K_1) z^{-2} } .
        $$

    References:
        - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Appendix C: Phase Locked Loops.

    Examples:
        See the :ref:`phase-locked-loop` example.

    Group:
        pll
    """

    def __init__(
        self, noise_bandwidth: float, damping_factor: float, K0: float = 1.0, Kp: float = 1.0, sample_rate: float = 1.0
    ):
        r"""
        Creates a closed-loop PLL analysis object.

        Arguments:
            noise_bandwidth: The normalized noise bandwidth $B_n T$ of the loop filter,
                where $B_n$ is the noise bandwidth in Hz and $T$ is the sampling period in seconds.
            damping_factor: The damping factor $\zeta$ of the loop filter. $\zeta = 1$ is critically damped,
                $\zeta < 1$ is underdamped, and $\zeta > 1$ is overdamped.
            K0: The NCO gain $K_0$.
            Kp: The gain $K_p$ of the phase error detector (PED) or time error detector (TED).
            sample_rate: The sample rate $f_s$ of the PLL in Hz.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        lf = LoopFilter(noise_bandwidth, damping_factor, K0, Kp)
        K1 = lf.K1
        K2 = lf.K2

        # b0 = 0
        b1 = Kp * K0 * (K1 + K2)
        b2 = -Kp * K0 * K1

        a0 = 1
        a1 = -2 * (1 - 0.5 * Kp * K0 * (K1 + K2))
        a2 = 1 - Kp * K0 * K1

        # Create an IIR filter that represents the closed-loop transfer function of the PLL
        # self._iir = IIR([b0, b1, b2], [a0, a1, a2])
        self._iir = IIR([b1, b2], [a0, a1, a2])  # Suppress warning about leading zero

        self._sample_rate = sample_rate
        self._BnT = noise_bandwidth
        self._zeta = damping_factor

        self._K0 = K0
        self._Kp = Kp
        self._K1 = K1
        self._K2 = K2

    def phase_lock_time(self) -> float:
        r"""
        Returns the phase lock time of the PLL.

        $$
        T_{PL} = \frac{1.3}{B_n}
        $$

        Returns:
            The time $T_{PL}$ it takes the PLL to lock onto the input signal's phase in seconds.

        References:
            - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Equation C.40.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return 1.3 / self.Bn

    def frequency_lock_time(self, freq_offset: float) -> float:
        r"""
        Returns the frequency lock time of the PLL.

        $$
        T_{FL} = 4 \frac{(\Delta f)^2}{B_n^3}
        $$

        Arguments:
            freq_offset: The frequency offset $\Delta f$ of the input signal in Hz.

        Returns:
            The time $T_{FL}$ it takes the PLL to lock onto the input signal's frequency in seconds.

        References:
            - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Equation C.39.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return 4 * freq_offset**2 / self.Bn**3

    def lock_time(self, freq_offset: float) -> float:
        r"""
        Returns the lock time of the PLL.

        $$
        T_{LOCK} = T_{PL} + T_{FL} = \frac{1.3}{B_n} + 4 \frac{(\Delta f)^2}{B_n^3}
        $$

        Arguments:
            freq_offset: The frequency offset $\Delta f$ of the input signal in Hz.

        Returns:
            The time $T_{LOCK}$ it takes the PLL to lock onto the input signal's phase and frequency in seconds.

        References:
            - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Equation C.38.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self.phase_lock_time() * self.frequency_lock_time(freq_offset)

    def phase_error_variance(self, cn0: float) -> float:
        r"""
        Returns the variance of the phase error of the PLL in steady state.

        $$
        \sigma_{\theta_e}^2 = \frac{N_0 B_n}{C}
        $$

        Arguments:
            cn0: The carrier-to-noise density ratio $C/N_0$ of the input signal in dB-Hz.

        Returns:
            The variance of the phase error $\sigma_{\theta_e}^2$ of the PLL in radians^2.

        References:
            - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Equation C.43.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        cn0_linear = linear(cn0)
        return self.Bn / cn0_linear

    @property
    def sample_rate(self) -> float:
        """
        The sample rate $f_s$ of the PLL in samples/s.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._sample_rate

    @property
    def BnT(self) -> float:
        """
        The normalized noise bandwidth $B_n T$ of the PLL.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._BnT

    @property
    def Bn(self) -> float:
        """
        The noise bandwidth $B_n$ of the PLL in Hz.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self.BnT * self.sample_rate

    @property
    def zeta(self) -> float:
        r"""
        The damping factor $\zeta$ of the PLL.

        A damping factor of 1 is critically damped, less than 1 is underdamped, and greater than 1 is overdamped.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._zeta

    @property
    def K0(self) -> float:
        """
        The NCO gain $K_0$.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._K0

    @property
    def Kp(self) -> float:
        """
        The phase error detector (PED) gain $K_p$.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._Kp

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
        The IIR filter that represents the closed-loop transfer function of the PLL.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self._iir

    @property
    def omega_n(self) -> float:
        r"""
        The natural frequency $\omega_n$ of the PLL in radians/s.

        References:
            - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Equation C.33.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return np.sqrt(self.K0 * self.Kp * self.K2)

    @property
    def omega_3dB(self) -> float:
        r"""
        The 3-dB bandwidth $\omega_{3\textrm{dB}}$ of the PLL in radians/s.

        References:
            - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Equation C.34.

        Examples:
            See the :ref:`phase-locked-loop` example.
        """
        return self.omega_n * np.sqrt(1 + 2 * self.zeta**2 + np.sqrt((1 + 2 * self.zeta**2) ** 2 + 1))
