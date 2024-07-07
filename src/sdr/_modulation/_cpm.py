"""
A module containing a base class for continuous phase modulations.
"""

from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._filter import Decimator, Interpolator
from .._helper import export, verify_arraylike, verify_equation, verify_literal, verify_scalar
from .._nco import NCO
from .._sequence import binary_code, gray_code
from ._pulse_shapes import rectangular


@export
class CPM:
    r"""
    Implements continuous-phase modulation (CPM).

    .. nomenclature::
        :collapsible:

        - $k$: Symbol index
        - $n$: Sample index
        - $s[k]$: Decimal symbols
        - $a[k]$ Complex symbols
        - $x[n]$: Pulse-shaped complex samples
        - $\tilde{x}[n]$: Received (noisy) pulse-shaped complex samples
        - $\tilde{a}[k]$: Received (noisy) complex symbols
        - $\hat{a}[k]$: Complex symbol decisions
        - $\hat{s}[k]$: Decimal symbol decisions

    Group:
        modulation-continuous-phase
    """

    def __init__(
        self,
        order: int,
        index: float = 0.5,
        symbol_labels: Literal["bin", "gray"] | npt.ArrayLike = "bin",
        phase_offset: float = 0.0,
        sps: int = 8,
        # pulse_shape: npt.ArrayLike | Literal["rect", "rc", "srrc", "gaussian"] = "rect",
        pulse_shape: npt.ArrayLike | Literal["rect"] = "rect",
        span: int = 1,
        # alpha: float | None = None,
        # time_bandwidth: float | None = None,
    ):
        r"""
        Creates a new continuous-phase modulation object.

        Arguments:
            order: The modulation order $M = 2^k$.
            index: The modulation index $h$. The modulation index is the ratio of the frequency deviation to the
                symbol rate $h = \Delta f / f_{sym}$. The phase change per symbol is $\pi h$.
            symbol_labels: The decimal symbol labels of consecutive complex symbols.

                - `"bin"`: The symbols are binary-coded. Adjacent symbols may differ by more than one bit.
                - `"gray":` The symbols are Gray-coded. Adjacent symbols only differ by one bit.
                - `npt.ArrayLike`: An $M$-length array whose indices are the default symbol labels and whose values are
                  the new symbol labels.

            phase_offset: A phase offset $\phi$ in degrees.
            sps: The number of samples per symbol $f_s / f_{sym}$.
            pulse_shape: The pulse shape $h[n]$ of the instantaneous frequency of the signal. If a string is passed,
                the pulse shape is normalized such that the maximum value is 1.

                - `npt.ArrayLike`: A custom pulse shape. It is important that `sps` matches the design
                  of the pulse shape. See :ref:`pulse-shaping-functions`.
                - `"rect"`: Rectangular pulse shape.

            span: The span of the pulse shape in symbols. This is only used if `pulse_shape` is a string.

        See Also:
            sdr.rectangular
        """
        self._order = verify_scalar(order, int=True, inclusive_min=2, power_of_two=True)  # Modulation order
        self._bps = int(np.log2(self._order))  # Coded bits per symbol
        self._index = verify_scalar(index, float=True, non_negative=True)  # Modulation index

        if isinstance(symbol_labels, str):
            verify_literal(symbol_labels, ["bin", "gray"])
            if symbol_labels == "bin":
                self._symbol_labels = binary_code(self.bps)
                self._symbol_labels_str = "bin"
            elif symbol_labels == "gray":
                self._symbol_labels = gray_code(self.bps)
                self._symbol_labels_str = "gray"
        else:
            symbol_labels = verify_arraylike(symbol_labels, int=True, ndim=1, size=self.order)
            verify_equation(np.unique(symbol_labels).size == self.order)
            self._symbol_labels = symbol_labels
            self._symbol_labels_str = self._symbol_labels

        self._phase_offset = verify_scalar(phase_offset, float=True)  # Phase offset in degrees
        self._sps = verify_scalar(sps, int=True, positive=True)  # Samples per symbol
        verify_scalar(span, int=True, positive=True)

        if isinstance(pulse_shape, str):
            verify_literal(pulse_shape, ["rect"])
            if pulse_shape == "rect":
                self._pulse_shape = rectangular(self.sps, span=span, norm="passband") / 2
                # self._pulse_shape = np.ones(self.sps * span) / (self.sps * span) / 2
            # elif pulse_shape == "sine":
            #     # self._pulse_shape = half_sine(self.sps, norm="passband") / 2
            #     self._pulse_shape = 1 - np.cos(2 * np.pi * np.arange(0.5, self.sps + 0.5, 1) / self.sps)
            #     self._pulse_shape = _normalize(self._pulse_shape, norm="passband") / 2
            # elif pulse_shape == "rc":
            #     if alpha is None:
            #         alpha = 0.2
            #     self._pulse_shape = raised_cosine(alpha, span, self.sps, norm="passband") / 2
            # elif pulse_shape == "gaussian":
            #     if time_bandwidth is None:
            #         time_bandwidth = 0.3
            #     self._pulse_shape = gaussian(time_bandwidth, span, self.sps, norm="passband") / 2
        else:
            self._pulse_shape = verify_arraylike(pulse_shape, float=True, ndim=1)  # Pulse shape

        # if alpha is not None and pulse_shape not in ["rc", "srrc"]:
        #     raise ValueError("Argument 'alpha' is only valid for 'rc' and 'srrc' pulse shapes, not {pulse_shape!r}.")
        # if time_bandwidth is not None and pulse_shape not in ["gaussian"]:
        #     raise ValueError("Argument 'time_bandwidth' is only valid for 'gaussian' pulse shape, not {pulse_shape!r}.")

        self._tx_filter = Interpolator(self.sps, self.pulse_shape)  # Transmit pulse shaping filter
        self._rx_filter = Decimator(self.sps, self.pulse_shape[::-1].conj())  # Receive matched filter

        self._nco = NCO()

    def __repr__(self) -> str:
        return f"sdr.{type(self).__name__}({self.order}, {self.index}, phase_offset={self.phase_offset})"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  index: {self.index}"
        string += f"\n  phase_offset: {self.phase_offset}"
        return string

    def modulate(self, s: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        r"""
        Modulates the decimal symbols $s[k]$ into pulse-shaped complex samples $x[n]$.

        Arguments:
            s: The decimal symbols $s[k]$ to modulate, $0$ to $M-1$.

        Returns:
            The pulse-shaped complex samples $x[n]$ with :obj:`sps` samples per symbol
            and length `sps * s.size + pulse_shape.size - 1`.
        """
        s = verify_arraylike(s, int=True)  # Decimal symbols
        return self._modulate(s)

    def _modulate(self, s: npt.NDArray[np.int_]) -> npt.NDArray[np.complex128]:
        s = self._symbol_labels[s]  # Relabeled decimal symbols
        freq = self.index * (2 * s - (self.order - 1))  # Instantaneous frequency
        print(freq)
        # return f
        freq_ps = self._tx_pulse_shape(freq)  # Pulse-shaped instantaneous frequency
        print(freq_ps)
        # phase_ps = np.cumsum(freq_ps)  # Pulse-shaped instantaneous phase
        # phase_ps = np.insert(phase_ps, 0, 0)  # Start with phase 0
        # phase_ps = phase_ps[:-1]  # Trim last phase
        # # return phase_ps
        # x = np.exp(1j * np.pi / self.sps * phase_ps)  # Complex samples
        # # x = np.exp(1j * (2 * np.pi / self.sps * self.index * freq_ps + self._phase_offset))  # Complex samples
        x = self._nco(2 * np.pi * freq_ps, output="complex-exp")  # Complex samples
        return x

    def _tx_pulse_shape(self, a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        x = self._tx_filter(a, mode="full")  # Complex samples
        return x

    def demodulate(self, x_tilde: npt.ArrayLike) -> npt.NDArray[np.int_]:
        r"""
        Demodulates the pulse-shaped complex samples into decimal symbol decisions.

        This method applies matched filtering and maximum-likelihood estimation.

        Arguments:
            x_tilde: The received pulse-shaped complex samples $\tilde{x}[n]$ to demodulate, with :obj:`sps`
                samples per symbol and length `sps * s_hat.size + pulse_shape.size - 1`.

        Returns:
            - The decimal symbol decisions $\hat{s}[k]$, $0$ to $M-1$.
        """
        x_tilde = verify_arraylike(x_tilde, complex=True)  # Complex samples
        return self._demodulate(x_tilde)

    def _demodulate(self, x_tilde: npt.NDArray[np.complex128]) -> npt.NDArray[np.int_]:
        raise NotImplementedError("Demodulation for continuous-phase modulations is not supported.")

    @abc.abstractmethod
    def ber(self, ebn0: npt.ArrayLike | None = None) -> np.ndarray:
        r"""
        Computes the bit error rate (BER) at the provided $E_b/N_0$ values.

        Arguments:
            ebn0: Bit energy $E_b$ to noise PSD $N_0$ ratio in dB.

        Returns:
            The bit error rate $P_b$.

        See Also:
            sdr.esn0_to_ebn0, sdr.snr_to_ebn0
        """
        raise NotImplementedError("Bit error rate calculation for continuous-phase modulations is not supported.")

    @abc.abstractmethod
    def ser(self, esn0: npt.ArrayLike) -> np.ndarray:
        r"""
        Computes the symbol error rate (SER) at the provided $E_s/N_0$ values.

        Arguments:
            esn0: Symbol energy $E_s$ to noise PSD $N_0$ ratio in dB.

        Returns:
            The symbol error rate $P_e$.

        See Also:
            sdr.ebn0_to_esn0, sdr.snr_to_esn0
        """
        raise NotImplementedError("Symbol error rate calculation for continuous-phase modulations is not supported.")

    @property
    def order(self) -> int:
        r"""
        The modulation order $M = 2^k$.
        """
        return self._order

    @property
    def bps(self) -> int:
        r"""
        The number of coded bits per symbol $k = \log_2 M$.
        """
        return self._bps

    @property
    def index(self) -> float:
        r"""
        The modulation index $h$.

        The modulation index is the ratio of the frequency deviation to the symbol rate $h = \Delta f / f_{sym}$.
        The phase change per symbol is $\pi h$.
        """
        return self._index

    @property
    def phase_offset(self) -> float:
        r"""
        The phase offset $\phi$ in degrees.
        """
        return self._phase_offset

    @property
    def sps(self) -> int:
        r"""
        The number of samples per symbol $f_s / f_{sym}$.
        """
        return self._sps

    @property
    def pulse_shape(self) -> np.ndarray:
        r"""
        The pulse shape $h[n]$ of the instantaneous frequency of the signal.
        """
        return self._pulse_shape

    @property
    def tx_filter(self) -> Interpolator:
        r"""
        The transmit interpolating pulse shaping filter. The filter coefficients are the pulse shape $h[n]$.
        """
        return self._tx_filter

    @property
    def rx_filter(self) -> Decimator:
        r"""
        The receive decimating matched filter. The filter coefficients are matched to the pulse shape $h[-n]^*$.
        """
        return self._rx_filter
