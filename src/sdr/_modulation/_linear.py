"""
A module containing a base class for linear phase/amplitude modulations.
"""

from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._filter import Decimator, Interpolator
from .._helper import convert_output, export, verify_arraylike, verify_literal, verify_scalar
from ._pulse_shapes import raised_cosine, rectangular, root_raised_cosine


@export
class LinearModulation:
    r"""
    Implements linear phase/amplitude modulation with arbitrary symbol mapping.

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
        modulation-linear
    """

    def __init__(
        self,
        symbol_map: npt.ArrayLike,
        phase_offset: float = 0.0,
        sps: int = 8,
        pulse_shape: npt.ArrayLike | Literal["rect", "rc", "srrc"] = "rect",
        span: int | None = None,
        alpha: float | None = None,
    ):
        r"""
        Creates a new linear phase/amplitude modulation object.

        Arguments:
            symbol_map: The symbol mapping $\{0, \dots, M-1\} \mapsto \mathbb{C}$. An $M$-length array whose indices
                are decimal symbols $s[k]$ and whose values are complex symbols $a[k]$, where $M$ is the
                modulation order.
            phase_offset: A phase offset $\phi$ in degrees to apply to `symbol_map`.
            sps: The number of samples per symbol $f_s / f_{sym}$.
            pulse_shape: The pulse shape $h[n]$ of the modulated signal.

                - `npt.ArrayLike`: A custom pulse shape. It is important that `sps` matches the design
                  of the pulse shape. See :ref:`pulse-shaping-functions`.
                - `"rect"`: Rectangular pulse shape.
                - `"rc"`: Raised cosine pulse shape.
                - `"srrc"`: Square-root raised cosine pulse shape.

            span: The span of the pulse shape in symbols. This is only used if `pulse_shape` is a string.
                If `None`, 1 is used for `"rect"` and 10 is used for `"rc"` and `"srrc"`.
            alpha: The roll-off factor of the pulse shape. If `None`, 0.2 is used for `"rc"` and `"srrc"`.

        See Also:
            sdr.rectangular, sdr.raised_cosine, sdr.root_raised_cosine
        """
        self._symbol_map = verify_arraylike(symbol_map, complex=True, ndim=1)  # Decimal-to-complex symbol map
        self._order = verify_scalar(self._symbol_map.size, power_of_two=True)  # Modulation order
        self._bps = int(np.log2(self._order))  # Coded bits per symbol
        self._phase_offset = verify_scalar(phase_offset, float=True)  # Phase offset in degrees
        self._sps = verify_scalar(sps, int=True, positive=True)  # Samples per symbol

        if isinstance(pulse_shape, str):
            verify_literal(pulse_shape, ["rect", "rc", "srrc"])
            if pulse_shape == "rect":
                if span is None:
                    span = 1
                self._pulse_shape = rectangular(self.sps, span=span)
            elif pulse_shape == "rc":
                if span is None:
                    span = 10
                if alpha is None:
                    alpha = 0.2
                self._pulse_shape = raised_cosine(alpha, span, self.sps)
            elif pulse_shape == "srrc":
                if span is None:
                    span = 10
                if alpha is None:
                    alpha = 0.2
                self._pulse_shape = root_raised_cosine(alpha, span, self.sps)
        else:
            self._pulse_shape = verify_arraylike(pulse_shape, float=True, ndim=1)  # Pulse shape

        self._tx_filter = Interpolator(self.sps, self.pulse_shape)  # Transmit pulse shaping filter
        self._rx_filter = Decimator(self.sps, self.pulse_shape[::-1].conj())  # Receive matched filter

    def __repr__(self) -> str:
        return f"sdr.{type(self).__name__}({self.symbol_map.tolist()}, phase_offset={self.phase_offset})"

    def __str__(self) -> str:
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  symbol_map: {self.symbol_map.shape} shape"
        string += f"\n    {self.symbol_map.tolist()}"
        string += f"\n  phase_offset: {self.phase_offset}"
        return string

    def map_symbols(self, s: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        r"""
        Converts the decimal symbols into complex symbols.

        Arguments:
            s: The decimal symbols $s[k]$ to map, $0$ to $M-1$.

        Returns:
            The complex symbols $a[k]$.
        """
        s = verify_arraylike(s, int=True)  # Decimal symbols
        a = self._map_symbols(s)  # Complex symbols
        return convert_output(a)

    def _map_symbols(self, s: npt.NDArray[np.int_]) -> npt.NDArray[np.complex128]:
        a = self.symbol_map[s]  # Complex symbols
        return a

    def decide_symbols(self, a_tilde: npt.ArrayLike) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.complex128]]:
        r"""
        Converts the received complex symbols into MLE symbol decisions.

        This method uses maximum-likelihood estimation (MLE).

        Arguments:
            a_tilde: The received complex symbols $\tilde{a}[k]$.

        Returns:
            - The decimal symbol decisions $\hat{s}[k]$, $0$ to $M-1$.
            - The complex symbol decisions $\hat{a}[k]$.
        """
        a_tilde = verify_arraylike(a_tilde, complex=True)  # Complex symbols
        s_hat, a_hat = self._decide_symbols(a_tilde)  # Decimal and complex symbol decisions
        return convert_output(s_hat), convert_output(a_hat)

    def _decide_symbols(
        self, a_tilde: npt.NDArray[np.complex128]
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.complex128]]:
        error_vectors = np.subtract.outer(a_tilde, self.symbol_map)
        s_hat = np.argmin(np.abs(error_vectors), axis=-1)
        a_hat = self.symbol_map[s_hat]
        return s_hat, a_hat

    def modulate(self, s: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        r"""
        Modulates the decimal symbols into pulse-shaped complex samples.

        Arguments:
            s: The decimal symbols $s[k]$ to modulate, $0$ to $M-1$.

        Returns:
            The pulse-shaped complex samples $x[n]$ with :obj:`sps` samples per symbol
            and length `sps * s.size + pulse_shape.size - 1`.
        """
        s = verify_arraylike(s, int=True)  # Decimal symbols
        return self._modulate(s)

    def _modulate(self, s: npt.NDArray[np.int_]) -> npt.NDArray[np.complex128]:
        a = self._map_symbols(s)  # Complex symbols
        x = self._tx_pulse_shape(a)  # Complex samples
        return x

    def _tx_pulse_shape(self, a: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        x = self._tx_filter(a, mode="full")  # Complex samples
        return x

    def demodulate(
        self,
        x_tilde: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        r"""
        Demodulates the pulse-shaped complex samples.

        This method uses matched filtering and maximum-likelihood estimation.

        Arguments:
            x_tilde: The received pulse-shaped complex samples $\tilde{x}[n]$ to demodulate, with :obj:`sps`
                samples per symbol and length `sps * s_hat.size + pulse_shape.size - 1`.

        Returns:
            - The decimal symbol decisions $\hat{s}[k]$, $0$ to $M-1$.
            - The matched filter outputs $\tilde{a}[k]$.
            - The complex symbol decisions $\hat{a}[k]$.
        """
        x_tilde = verify_arraylike(x_tilde, complex=True)  # Complex samples
        # Decimal symbol decisions, complex symbols, complex symbol decisions
        s_hat, a_tilde, a_hat = self._demodulate(x_tilde)
        return convert_output(s_hat), convert_output(a_tilde), convert_output(a_hat)

    def _demodulate(
        self,
        x_tilde: npt.NDArray[np.complex128],
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        a_tilde = self._rx_matched_filter(x_tilde)  # Complex symbols
        s_hat, a_hat = self._decide_symbols(a_tilde)  # Decimal and complex symbol decisions
        return s_hat, a_tilde, a_hat

    def _rx_matched_filter(
        self,
        x_tilde: npt.NDArray[np.complex128],
    ) -> npt.NDArray[np.complex128]:
        if self.pulse_shape.size % self.sps == 0:
            x_tilde = np.insert(x_tilde, 0, 0)

        a_tilde = self._rx_filter(x_tilde, mode="full")  # Complex symbols

        span = self.pulse_shape.size // self.sps
        if span == 1:
            N_symbols = x_tilde.size // self.sps
            offset = span
        else:
            N_symbols = x_tilde.size // self.sps - span
            offset = span

        # Select the symbol decisions from the output of the decimating filter
        a_tilde = a_tilde[offset : offset + N_symbols]

        return a_tilde

    @abc.abstractmethod
    def ber(self, ebn0: npt.ArrayLike) -> npt.NDArray[np.float64]:
        r"""
        Computes the bit error rate (BER) at the provided $E_b/N_0$ values.

        Arguments:
            ebn0: Bit energy $E_b$ to noise PSD $N_0$ ratio in dB.

        Returns:
            The bit error rate $P_b$.

        See Also:
            sdr.esn0_to_ebn0, sdr.snr_to_ebn0
        """
        raise NotImplementedError("Bit error rate calculation for arbitrary linear modulations is not supported.")

    @abc.abstractmethod
    def ser(self, esn0: npt.ArrayLike) -> npt.NDArray[np.float64]:
        r"""
        Computes the symbol error rate (SER) at the provided $E_s/N_0$ values.

        Arguments:
            esn0: Symbol energy $E_s$ to noise PSD $N_0$ ratio in dB.

        Returns:
            The symbol error rate $P_e$.

        See Also:
            sdr.ebn0_to_esn0, sdr.snr_to_esn0
        """
        raise NotImplementedError("Symbol error rate calculation for arbitrary linear modulations is not supported.")

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
    def phase_offset(self) -> float:
        r"""
        The phase offset $\phi$ in degrees.
        """
        return self._phase_offset

    @property
    def symbol_map(self) -> npt.NDArray[np.complex128]:
        r"""
        The symbol map $\{0, \dots, M-1\} \mapsto \mathbb{C}$.

        This maps decimal symbols from $0$ to $M-1$ to complex symbols.
        """
        return self._symbol_map

    @property
    def sps(self) -> int:
        r"""
        The number of samples per symbol $f_s / f_{sym}$.
        """
        return self._sps

    @property
    def pulse_shape(self) -> npt.NDArray[np.float64]:
        r"""
        The pulse shape $h[n]$ of the modulated signal.
        """
        return self._pulse_shape

    @property
    def tx_filter(self) -> Interpolator:
        r"""
        The transmit interpolating pulse shaping filter.

        The filter coefficients are the pulse shape $h[n]$.
        """
        return self._tx_filter

    @property
    def rx_filter(self) -> Decimator:
        r"""
        The receive decimating matched filter.

        The filter coefficients are matched to the pulse shape $h[-n]^*$.
        """
        return self._rx_filter
