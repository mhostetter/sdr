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
    """
    Implements linear phase/amplitude modulation with arbitrary symbol mapping.

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
        symbol_map = np.asarray(symbol_map)
        if not symbol_map.ndim == 1:
            raise ValueError(f"Argument 'symbol_map' must be 1-D, not {symbol_map.ndim}-D.")
        if not np.log2(symbol_map.size).is_integer():
            raise ValueError(f"Argument 'symbol_map' must have a size that is a power of 2, not {symbol_map.size}.")
        self._symbol_map = symbol_map  # Decimal-to-complex symbol map
        self._order = symbol_map.size  # Modulation order
        self._bps = int(np.log2(self._order))  # Bits per symbol

        if not isinstance(phase_offset, (int, float)):
            raise TypeError(f"Argument 'phase_offset' must be a number, not {type(phase_offset)}.")
        self._phase_offset = phase_offset  # Phase offset in degrees

        if not isinstance(sps, int):
            raise TypeError(f"Argument 'sps' must be an integer, not {type(sps)}.")
        if not sps > 1:
            raise ValueError(f"Argument 'sps' must be greater than 1, not {sps}.")
        self._sps = sps  # Samples per symbol

        if isinstance(pulse_shape, str):
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
                raise ValueError(f"Argument 'pulse_shape' must be 'rect', 'rc', or 'srrc', not {pulse_shape!r}.")
        else:
            pulse_shape = np.asarray(pulse_shape)
            if not pulse_shape.ndim == 1:
                raise ValueError(f"Argument 'pulse_shape' must be 1-D, not {pulse_shape.ndim}-D.")
            self._pulse_shape = pulse_shape  # Pulse shape

        self._tx_filter = Interpolator(self.sps, self.pulse_shape)  # Transmit pulse shaping filter
        self._rx_filter = Decimator(self.sps, self.pulse_shape[::-1].conj())  # Receive matched filter

    def __repr__(self) -> str:
        """
        Returns a code-styled string representation of the object.
        """
        return f"sdr.{type(self).__name__}({self.symbol_map.tolist()}, phase_offset={self.phase_offset})"

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the object.
        """
        string = f"sdr.{type(self).__name__}:"
        string += f"\n  order: {self.order}"
        string += f"\n  symbol_map: {self.symbol_map.shape} shape"
        string += f"\n    {self.symbol_map.tolist()}"
        string += f"\n  phase_offset: {self.phase_offset}"
        return string

    def map_symbols(self, s: npt.ArrayLike) -> np.ndarray:
        r"""
        Converts the decimal symbols $s[k]$ to complex symbols $a[k]$.

        Arguments:
            s: The decimal symbols $s[k]$ to map, $0$ to $M-1$.

        Returns:
            The complex symbols $a[k]$.
        """
        return self._map_symbols(s)

    def _map_symbols(self, s: npt.ArrayLike) -> np.ndarray:
        s = np.asarray(s)  # Decimal symbols
        a = self.symbol_map[s]  # Complex symbols
        return a

    def decide_symbols(self, a_hat: npt.ArrayLike) -> np.ndarray:
        r"""
        Converts the received complex symbols $\hat{a}[k]$ into decimal symbol decisions $\hat{s}[k]$
        using maximum-likelihood estimation (MLE).

        Arguments:
            a_hat: The received complex symbols $\hat{a}[k]$.

        Returns:
            The decimal symbol decisions $\hat{s}[k]$, $0$ to $M-1$.
        """
        return self._decide_symbols(a_hat)

    def _decide_symbols(self, a_hat: npt.ArrayLike) -> np.ndarray:
        error_vectors = np.subtract.outer(a_hat, self.symbol_map)
        s_hat = np.argmin(np.abs(error_vectors), axis=-1)
        return s_hat

    def modulate(self, s: npt.ArrayLike) -> np.ndarray:
        r"""
        Modulates the decimal symbols $s[k]$ into pulse-shaped complex samples $x[n]$.

        Arguments:
            s: The decimal symbols $s[k]$ to modulate, $0$ to $M-1$.

        Returns:
            The pulse-shaped complex samples $x[n]$ with :obj:`sps` samples per symbol
            and length `sps * s.size + pulse_shape.size - 1`.
        """
        return self._modulate(s)

    def _modulate(self, s: npt.ArrayLike) -> np.ndarray:
        a = self._map_symbols(s)  # Complex symbols
        x = self._tx_pulse_shape(a)  # Complex samples
        return x

    def _tx_pulse_shape(self, a: npt.ArrayLike) -> np.ndarray:
        a = np.asarray(a)  # Complex symbols
        x = self._tx_filter(a, mode="full")  # Complex samples
        return x

    def demodulate(self, x_hat: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Demodulates the pulse-shaped complex samples $\hat{x}[n]$ into decimal symbol decisions $\hat{s}[k]$
        using matched filtering and maximum-likelihood estimation.

        Arguments:
            x_hat: The received pulse-shaped complex samples $\hat{x}[n]$ to demodulate, with :obj:`sps`
                samples per symbol and length `sps * s_hat.size + pulse_shape.size - 1`.

        Returns:
            - The decimal symbol decisions $\hat{s}[k]$, $0$ to $M-1$.
            - The matched filter outputs $\hat{a}[k]$.
        """
        return self._demodulate(x_hat)

    def _demodulate(self, x_hat: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        x_hat = np.asarray(x_hat)  # Complex samples
        a_hat = self._rx_matched_filter(x_hat)  # Complex symbols
        s_hat = self._decide_symbols(a_hat)  # Decimal symbols
        return s_hat, a_hat

    def _rx_matched_filter(self, x_hat: npt.ArrayLike) -> np.ndarray:
        x_hat = np.asarray(x_hat)  # Complex samples

        if self.pulse_shape.size % self.sps == 0:
            x_hat = np.insert(x_hat, 0, 0)

        a_hat = self._rx_filter(x_hat, mode="full")  # Complex symbols

        span = self.pulse_shape.size // self.sps
        if span == 1:
            N_symbols = x_hat.size // self.sps
            offset = span
        else:
            N_symbols = x_hat.size // self.sps - span
            offset = span

        # Select the symbol decisions from the output of the decimating filter
        a_hat = a_hat[offset : offset + N_symbols]

        return a_hat

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
        raise NotImplementedError("Bit error rate calculation for arbitrary linear modulations is not supported.")

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
        The number of bits per symbol $k = \log_2 M$.
        """
        return self._bps

    @property
    def phase_offset(self) -> float:
        r"""
        The phase offset $\phi$ in degrees.
        """
        return self._phase_offset

    @property
    def symbol_map(self) -> np.ndarray:
        r"""
        The symbol map $\{0, \dots, M-1\} \mapsto \mathbb{C}$. This maps decimal symbols from $0$ to $M-1$
        to complex symbols.
        """
        return self._symbol_map

    @property
    def sps(self) -> int:
        r"""
        The number of samples per symbol $f_s / f_{sym}$.
        """
        return self._sps

    @property
    def pulse_shape(self) -> np.ndarray:
        r"""
        The pulse shape $h[n]$ of the modulated signal.
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
