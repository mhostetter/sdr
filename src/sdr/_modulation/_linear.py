"""
A module containing a base class for linear phase/amplitude modulations.
"""

from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal

from .._filter import Decimator, Interpolator
from .._helper import export
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
        samples_per_symbol: int = 8,
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
            samples_per_symbol: The number of samples per symbol $f_s / f_{sym}$.
            pulse_shape: The pulse shape $h[n]$ of the modulated signal.

                - `npt.ArrayLike`: A custom pulse shape. It is important that `samples_per_symbol` matches the design
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
        symbol_map = np.asarray(symbol_map)
        if not symbol_map.ndim == 1:
            raise ValueError(f"Argument 'symbol_map' must be 1-D, not {symbol_map.ndim}-D.")
        if not np.log2(symbol_map.size).is_integer():
            raise ValueError(f"Argument 'symbol_map' must have a size that is a power of 2, not {symbol_map.size}.")
        self._symbol_map = symbol_map  # Decimal-to-complex symbol map
        self._order = symbol_map.size  # Modulation order
        self._bits_per_symbol = int(np.log2(self._order))  # Coded bits per symbol

        if not isinstance(phase_offset, (int, float)):
            raise TypeError(f"Argument 'phase_offset' must be a number, not {type(phase_offset)}.")
        self._phase_offset = phase_offset  # Phase offset in degrees

        if not isinstance(samples_per_symbol, int):
            raise TypeError(f"Argument 'samples_per_symbol' must be an integer, not {type(samples_per_symbol)}.")
        if not samples_per_symbol > 1:
            raise ValueError(f"Argument 'samples_per_symbol' must be greater than 1, not {samples_per_symbol}.")
        self._samples_per_symbol = samples_per_symbol  # Samples per symbol

        if isinstance(pulse_shape, str):
            if pulse_shape == "rect":
                if span is None:
                    span = 1
                self._pulse_shape = rectangular(self.samples_per_symbol, span=span)
            elif pulse_shape == "rc":
                if span is None:
                    span = 10
                if alpha is None:
                    alpha = 0.2
                self._pulse_shape = raised_cosine(alpha, span, self.samples_per_symbol)
            elif pulse_shape == "srrc":
                if span is None:
                    span = 10
                if alpha is None:
                    alpha = 0.2
                self._pulse_shape = root_raised_cosine(alpha, span, self.samples_per_symbol)
            else:
                raise ValueError(f"Argument 'pulse_shape' must be 'rect', 'rc', or 'srrc', not {pulse_shape!r}.")
        else:
            pulse_shape = np.asarray(pulse_shape)
            if not pulse_shape.ndim == 1:
                raise ValueError(f"Argument 'pulse_shape' must be 1-D, not {pulse_shape.ndim}-D.")
            self._pulse_shape = pulse_shape  # Pulse shape

        self._tx_filter = Interpolator(self.samples_per_symbol, self.pulse_shape)  # Transmit pulse shaping filter
        self._rx_filter = Decimator(self.samples_per_symbol, self.pulse_shape[::-1].conj())  # Receive matched filter

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
        s = np.asarray(s)  # Decimal symbols
        return self._map_symbols(s)

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
        a_tilde = np.asarray(a_tilde)  # Complex symbols
        return self._decide_symbols(a_tilde)

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
            The pulse-shaped complex samples $x[n]$ with :obj:`samples_per_symbol` samples per symbol
            and length `samples_per_symbol * s.size + pulse_shape.size - 1`.
        """
        s = np.asarray(s)  # Decimal symbols
        return self._modulate(s)

    def _modulate(self, s: npt.NDArray[np.int_]) -> npt.NDArray[np.complex128]:
        a = self._map_symbols(s)  # Complex symbols
        x = self._tx_pulse_shape(a)  # Complex samples
        return x

    def _tx_pulse_shape(self, a: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        x = self._tx_filter(a, mode="full")  # Complex samples
        return x

    def demodulate(
        self, x_tilde: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        r"""
        Demodulates the pulse-shaped complex samples.

        This method uses matched filtering and maximum-likelihood estimation.

        Arguments:
            x_tilde: The received pulse-shaped complex samples $\tilde{x}[n]$ to demodulate, with :obj:`samples_per_symbol`
                samples per symbol and length `samples_per_symbol * s_hat.size + pulse_shape.size - 1`.

        Returns:
            - The decimal symbol decisions $\hat{s}[k]$, $0$ to $M-1$.
            - The matched filter outputs $\tilde{a}[k]$.
            - The complex symbol decisions $\hat{a}[k]$.
        """
        x_tilde = np.asarray(x_tilde)  # Complex samples
        return self._demodulate(x_tilde)

    def _demodulate(
        self, x_tilde: npt.NDArray[np.complex128]
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
        a_tilde = self._rx_matched_filter(x_tilde)  # Complex symbols
        s_hat, a_hat = self._decide_symbols(a_tilde)  # Decimal symbols
        return s_hat, a_tilde, a_hat

    def _rx_matched_filter(self, x_tilde: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        if self.pulse_shape.size % self.samples_per_symbol == 0:
            x_tilde = np.insert(x_tilde, 0, 0)

        a_tilde = self._rx_filter(x_tilde, mode="full")  # Complex symbols

        span = self.pulse_shape.size // self.samples_per_symbol
        if span == 1:
            N_symbols = x_tilde.size // self.samples_per_symbol
            offset = span
        else:
            N_symbols = x_tilde.size // self.samples_per_symbol - span
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
    def bits_per_symbol(self) -> int:
        r"""
        The number of coded bits per symbol $k = \log_2 M$.
        """
        return self._bits_per_symbol

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
    def samples_per_symbol(self) -> int:
        r"""
        The number of samples per symbol $f_s / f_{sym}$.
        """
        return self._samples_per_symbol

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
