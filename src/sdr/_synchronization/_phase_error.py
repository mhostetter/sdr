"""
A module containing phase error detectors (PEDs).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export
from .._modulation import LinearModulation


@export
class PED:
    r"""
    Implements a heuristic phase error detector (PED).

    Notes:
        The heuristic data-aided PED computes the angle between the received complex symbols $\tilde{a}[k]$
        and the known transmitted complex symbols $a[k]$.

        $$\theta_{e,DA}[k] = \angle \left( \tilde{a}[k] \cdot a^*[k] \right)$$

        The heuristic decision-directed PED computes the angle between the received complex symbols $\tilde{a}[k]$
        and the complex symbol decisions $\hat{a}[k]$.

        $$\theta_{e,DD}[k] = \angle \left( \tilde{a}[k] \cdot \hat{a}^*[k] \right)$$

    References:
        - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Section 7.2.1.

    Examples:
        Compare the data-aided and decision-directed PEDs on QPSK modulation.

        .. ipython:: python

            qpsk = sdr.PSK(4)
            ped = sdr.PED()

        .. ipython:: python

            error, da_error = ped.data_aided_error(qpsk); \
            error, dd_error = ped.decision_directed_error(qpsk)

            @savefig sdr_PED_1.png
            plt.figure(); \
            plt.plot(error, da_error, label="Data-aided"); \
            plt.plot(error, dd_error, label="Decision-directed"); \
            plt.grid(True, linestyle="--"); \
            plt.legend(); \
            plt.xlabel("Phase of received symbols (radians)"); \
            plt.ylabel("Phase error (radians)"); \
            plt.title("Comparison of data-aided and decision-directed PEDs on QPSK");

        Observe that the slope of the phase error $K_p = 1$ is the same for both the data-aided and decision-directed
        PEDs. Also note that the unambiguous range of the data-aided PED is $[-\pi, \pi)$ and the
        decision-directed PED is $[-\pi/M, \pi/M)$.

    Group:
        synchronization-ped
    """

    def __init__(self) -> None:
        """
        Initializes the PED.
        """
        return

    def __call__(
        self, received: npt.NDArray[np.complex128], reference: npt.NDArray[np.complex128]
    ) -> npt.NDArray[np.float64]:
        r"""
        Detects the phase error.

        Arguments:
            received: The received complex symbols $\tilde{a}[k]$.
            reference: The reference complex symbols, either the known transmitted complex symbols $a[k]$
                or the complex symbols decisions $\hat{a}[k]$.

        Returns:
            The detected phase error $\theta_e[k]$ in radians.
        """
        received = np.asarray(received)
        reference = np.asarray(reference)
        reference = reference / np.abs(reference)  # Normalize the reference symbols
        return np.angle(received * reference.conj())

    def data_aided_error(
        self, modem: LinearModulation, n_points: int = 1000
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""
        Simulates the average phase error of the data-aided PED using the specified modulation scheme.

        Arguments:
            modem: The linear modulation scheme.
            n_points: The number of points in the simulation.

        Returns:
            - The true phase error in radians.
            - The detected phase error in radians.
        """
        return _data_aided_error(self, modem, n_points)

    def decision_directed_error(
        self, modem: LinearModulation, n_points: int = 1000
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""
        Simulates the average phase error of the decision-directed PED using the specified modulation scheme.

        Arguments:
            modem: The linear modulation scheme.
            n_points: The number of points in the simulation.

        Returns:
            - The true phase error in radians.
            - The detected phase error in radians.
        """
        return _decision_directed_error(self, modem, n_points)

    @property
    def gain(self) -> float:
        r"""
        The gain of the phase error detector $K_p$.
        """
        return 1.0


@export
class MLPED(PED):
    r"""
    Implements a maximum-likelihood phase error detector (ML-PED).

    Notes:
        The data-aided ML-PED is computed using the received complex symbols $\tilde{a}[k]$ and the known
        transmitted complex symbols $a[k]$.

        $$\theta_{e,DA}[k] = \Im(\tilde{a}[k]) \cdot \Re(a[k]) - \Re(\tilde{a}[k]) \cdot \Im(a[k])) $$

        The data-aided ML-PED is computed using the received complex symbols $\tilde{a}[k]$ and the complex
        symbol decisions $\hat{a}[k]$.

        $$\theta_{e,DA}[k] = \Im(\tilde{a}[k]) \cdot \Re(\hat{a}[k]) - \Re(\tilde{a}[k]) \cdot \Im(\hat{a}[k]) $$

    References:
        - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Section 7.2.2.

    Examples:
        Compare the data-aided and decision-directed ML-PEDs on QPSK modulation.

        .. ipython:: python

            qpsk = sdr.PSK(4)
            A_rx, A_ref = 5, 3
            ped = sdr.MLPED(A_rx, A_ref)

        .. ipython:: python

            error, da_error = ped.data_aided_error(qpsk); \
            error, dd_error = ped.decision_directed_error(qpsk)

            @savefig sdr_MLPED_1.png
            plt.figure(); \
            plt.plot(error, da_error, label="Data-aided"); \
            plt.plot(error, dd_error, label="Decision-directed"); \
            plt.grid(True, linestyle="--"); \
            plt.legend(); \
            plt.xlabel("Phase of received symbols (radians)"); \
            plt.ylabel("Phase error (radians)"); \
            plt.title("Comparison of data-aided and decision-directed ML-PEDs on QPSK");

        Observe that the slope of the phase error $K_p = A_{rx,rms} A_{ref,rms}$ is the same for both the data-aided
        and decision-directed PEDs. It's very important to observe that the gain of the ML-PED is scaled by the
        received signal amplitude $A_{rx,rms}$ and the reference signal amplitude $A_{ref,rms}$. Because of this,
        the ML-PED should only be used with automatic gain control (AGC).

        .. ipython:: python

            ped.gain
            A_rx * A_ref

        Also note that the unambiguous range of the data-aided ML-PED is $[-\pi, \pi)$ and the
        decision-directed ML-PED is $[-\pi/M, \pi/M)$.

    Group:
        synchronization-ped
    """

    def __init__(self, A_received: float = 1, A_reference: float = 1) -> None:
        """
        Initializes the ML-PED.

        Arguments:
            A_received: The received signal RMS amplitude $A_{rx,rms}$.
            A_reference: The reference signal RMS amplitude $A_{ref,rms}$.
        """
        self._A_received = A_received
        self._A_reference = A_reference

    def __call__(self, received: npt.ArrayLike, reference: npt.ArrayLike) -> np.ndarray:
        received = np.asarray(received)
        reference = np.asarray(reference)
        return received.imag * reference.real - received.real * reference.imag

    def data_aided_error(
        self, modem: LinearModulation, n_points: int = 1000
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return _data_aided_error(self, modem, n_points, self._A_received, self._A_reference)

    def decision_directed_error(
        self, modem: LinearModulation, n_points: int = 1000
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return _decision_directed_error(self, modem, n_points, self._A_received, self._A_reference)

    @property
    def gain(self) -> float:
        return self.A_received * self.A_reference

    @property
    def A_received(self) -> float:
        """
        (Settable) The received signal RMS amplitude $A_{rx,rms}$.
        """
        return self._A_received

    @A_received.setter
    def A_received(self, A_received: float) -> None:
        self._A_received = A_received

    @property
    def A_reference(self) -> float:
        """
        (Settable) The reference signal RMS amplitude $A_{ref,rms}$.
        """
        return self._A_reference

    @A_reference.setter
    def A_reference(self, A_reference: float) -> None:
        self._A_reference = A_reference


def _data_aided_error(
    ped: PED, modem: LinearModulation, n_points: int = 1000, A_received: float = 1, A_reference: float = 1
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if not isinstance(ped, PED):
        raise TypeError(f"Argument 'ped' must be a PED, not {type(ped)}.")
    if not isinstance(modem, LinearModulation):
        raise TypeError(f"Argument 'modem' must be a LinearModulation, not {type(modem)}.")
    if not isinstance(n_points, int):
        raise TypeError(f"Argument 'n_points' must be an int, not {type(n_points)}.")

    error = np.linspace(-np.pi, np.pi, n_points)
    da_error = np.zeros(n_points, dtype=float)
    for reference in modem.symbol_map:
        received = reference * np.exp(1j * error)
        da_error += ped(A_received * received, A_reference * reference)
    da_error /= modem.order

    return error, da_error


def _decision_directed_error(
    ped: PED, modem: LinearModulation, n_points: int = 1000, A_received: float = 1, A_reference: float = 1
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if not isinstance(ped, PED):
        raise TypeError(f"Argument 'ped' must be a PED, not {type(ped)}.")
    if not isinstance(modem, LinearModulation):
        raise TypeError(f"Argument 'modem' must be a LinearModulation, not {type(modem)}.")
    if not isinstance(n_points, int):
        raise TypeError(f"Argument 'n_points' must be an int, not {type(n_points)}.")

    error = np.linspace(-np.pi, np.pi, n_points)
    dd_error = np.zeros(n_points, dtype=float)
    for reference in modem.symbol_map:
        received = reference * np.exp(1j * error)
        _, decided = modem.decide_symbols(received)
        dd_error += ped(A_received * received, A_reference * decided)
    dd_error /= modem.order

    return error, dd_error
