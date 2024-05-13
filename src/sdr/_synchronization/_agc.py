"""
A module for signal amplitude synchronization.
"""

from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt

from .._helper import export


@export
class AGC:
    r"""
    Implements an automatic gain controller (AGC).

    Notes:
        .. code-block:: text
            :caption: Automatic Gain Control Block Diagram

            x[n] -->X---------------------------------------------------------------------+--> y[n]
                    ^                                                                     |
                    |                                                                     |
                +--------+      +------+                     -1  +--------+   +-------+   |
                | exp(.) |<--+--| z^-1 |<--@<---X<--------@<-----| log(.) |<--|  |.|  |<--+
                +--------+   |  +------+   ^   alpha    log(R)   +--------+   +-------+
                             |             |  or beta
                             +-------------+

            x[n] = Input signal
            y[n] = Output signal
            alpha = Attack rate
            beta = Decay rate
            R = Reference magnitude
            @ = Adder
            X = Multiplier

    References:
        - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Section 9.5.
        - `Qasim Chaudhari, How Automatic Gain Control (AGC) Works.
          <https://wirelesspi.com/how-automatic-gain-control-agc-works/>`_

    Examples:
        Create an example received signal with two bursty signals surrounded by noise.

        .. ipython:: python

            x = np.exp(1j * 2 * np.pi * np.arange(5000) / 100); \
            x[0:1000] *= 0; \
            x[1000:2000] *= 10; \
            x[2000:3000] *= 0; \
            x[3000:4000] *= 0.1; \
            x[4000:5000] *= 0

            x += 0.001 * (np.random.randn(x.size) + 1j * np.random.randn(x.size))

            @savefig sdr_AGC_1.png
            plt.figure(); \
            sdr.plot.time_domain(x); \
            plt.title("Input signal");

        Create an AGC with an attack rate of $\alpha = 0.5$ and a decay rate of $\beta = 0.01$.
        Notice that over time the noise is amplified (according to the decay rate). Also notice that when the signal
        of interest appears the AGC gain is quickly decreased (according to the attack rate).

        .. ipython:: python

            agc = sdr.AGC(0.5, 0.01)
            y = agc(x)

            @savefig sdr_AGC_2.png
            plt.figure(); \
            sdr.plot.time_domain(y); \
            plt.ylim(-1.5, 1.5); \
            plt.title("Output signal");

    Group:
        synchronization-amplitude
    """

    def __init__(self, attack: float, decay: float, reference: float = 1.0, streaming: bool = False):
        r"""
        Creates an automatic gain controller (AGC).

        Arguments:
            attack: The attack rate $\alpha$. The attack rate is meant to attenuate strong signals.
                After $n_0 \approx 1 / \alpha$ samples the error will reduce to $1 / e$ of its original value.
            decay: The decay rate $\beta$. The decay rate is meant to amplify weak signals.
                After $n_0 \approx 1 / \beta$ samples the error will reduce to $1 / e$ of its original value.
            reference: The desired output magnitude.
            streaming: Indicates whether the AGC operates in streaming mode. In streaming mode, the gain is
                preserved between calls to :meth:`~AGC.__call__()`.
        """
        self._attack = attack
        self._decay = decay
        self._reference = reference
        self._streaming = streaming

        self._gain: float  # The linear gain state. Will be updated in reset().
        self.reset()

    def __call__(self, x: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Performs automatic gain control on the input signal.

        Arguments:
            x: The input signal $x[n]$.

        Returns:
            The output signal $y[n]$.
        """
        if not self.streaming:
            self.reset()

        y, self._gain = _numba_agc_loop(x, self.attack, self.decay, self.reference, self._gain)

        return y

    ##############################################################################
    # Streaming mode
    ##############################################################################

    def reset(self, gain: float = 1.0):
        """
        Resets the AGC gain. Only useful when using streaming mode.

        Arguments:
            gain: The initial linear gain of the AGC.

        Group:
            Streaming mode only
        """
        self._gain = gain

    @property
    def streaming(self) -> bool:
        """
        Indicates whether the AGC is in streaming mode.

        In streaming mode, the AGC gain is preserved between calls to :meth:`~AGC.__call__()`.

        Group:
            Streaming mode only
        """
        return self._streaming

    @property
    def gain(self) -> float:
        """
        The current linear gain.

        Group:
            Streaming mode only
        """
        return self._gain

    ##############################################################################
    # Properties
    ##############################################################################

    @property
    def attack(self) -> float:
        r"""
        (Settable) The attack rate $\alpha$.

        The attack rate is meant to attenuate strong signals. After $n_0 \approx 1 / \alpha$ samples the error will
        reduce to $1 / e$ of its original value.
        """
        return self._attack

    @attack.setter
    def attack(self, value: float):
        self._attack = value

    @property
    def decay(self) -> float:
        r"""
        (Settable) The decay rate $\beta$.

        The decay rate is meant to amplify weak signals. After $n_0 \approx 1 / \beta$ samples the error will reduce
        to $1 / e$ of its original value.
        """
        return self._decay

    @decay.setter
    def decay(self, value: float):
        self._decay = value

    @property
    def reference(self) -> float:
        """
        (Settable) The desired output magnitude.
        """
        return self._reference

    @reference.setter
    def reference(self, value: float):
        self._reference = value


@numba.jit(nopython=True, cache=True)
def _numba_agc_loop(
    x: npt.NDArray[np.complex128],
    attack: float,
    decay: float,
    reference: float,
    gain: float,
) -> tuple[npt.NDArray[np.complex128], float]:
    y = np.zeros_like(x)
    reference = np.log(reference)
    gain = np.log(gain)  # Convert to log domain

    for i in range(x.size):
        y[i] = x[i] * np.exp(gain)

        y_mag = np.abs(y[i])
        y_mag = max(y_mag, 1e-100)  # Don't let the magnitude go to zero
        y_mag = np.log(y_mag)

        error = reference - y_mag
        if error < 0:
            gain += attack * error
        else:
            gain += decay * error

    gain = np.exp(gain)  # Convert back to linear domain

    return y, gain
