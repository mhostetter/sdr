"""
A module containing a class for measuring bit error rates (BER) or symbol error rates (SER).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._helper import export


@export
class ErrorRate:
    r"""
    A class for measuring bit error rates (BER) or symbol error rates (SER).

    See Also:
        sdr.pack, sdr.unpack, sdr.hexdump

    .. _error-rate-example:

    Examples:
        Create a new bit error rate measurement object.

        .. ipython:: python

            ber = sdr.ErrorRate()
            x = [1, 1, 1, 1, 1]  # Reference bit vector

        Measure and accumulate bit errors from the first received bit vector containing 1 bit error at 10 dB SNR.
        The bit error rate of the first received bit vector is 0.2.

        .. ipython:: python

            x_hat = [1, 0, 1, 1, 1]; \
            ber.add(10, x, x_hat)

        Measure and accumulate bit errors from the second received bit vector containing 2 bit errors at 10 dB SNR.
        The bit error rate of the second received bit vector is 0.4.

        .. ipython:: python

            x_hat = [1, 0, 1, 0, 1]; \
            ber.add(10, x, x_hat)

        The total errors are 3, total bits 10, and average bit error rate 0.3.

        .. ipython:: python

            ber.errors(10), ber.counts(10), ber.error_rate(10)

        Average bit error rates for every SNR can be obtained as follows.

        .. ipython:: python

            ber.error_rates()

    Group:
        measurement-modulation
    """

    def __init__(self) -> None:
        """
        Creates a new error rate tabulation object.
        """
        self._errors: dict[float, int] = {}
        self._counts: dict[float, int] = {}

    def add(self, snr: float, x: npt.ArrayLike, x_hat: npt.ArrayLike) -> tuple[int, int, int]:
        r"""
        Measures the number of bit or symbol errors at the given signal-to-noise ratio (SNR).

        Arguments:
            snr: The signal-to-noise ratio (SNR) in dB. This can be $E_b/N_0$, $E_s/N_0$, $S/N$, $C/N_0$,
                or other SNR quantities. However, users are cautioned to be consistent for a given class instance.
            x: The transmitted bits or symbols $x[k]$.
            x_hat: The received bits or symbols $\hat{x}[k]$.

        Returns:
            - The number of errors.
            - The number of bits or symbols.
            - The error rate.

        See Also:
            sdr.pack, sdr.unpack

        Examples:
            See the class :ref:`error-rate-example` section.
        """
        x = np.asarray(x)
        x_hat = np.asarray(x_hat)
        if not x.shape == x_hat.shape:
            raise ValueError(f"Arguments 'x' and 'x_hat' must have the same shape, not {x.shape} and {x_hat.shape}.")

        errors = np.sum(x_hat != x)
        counts = x.size
        error_rate = errors / counts

        self._errors[snr] = self._errors.get(snr, 0) + errors
        self._counts[snr] = self._counts.get(snr, 0) + counts

        return errors, counts, error_rate

    def errors(self, snr: float) -> int:
        """
        Returns the number of errors at the specified signal-to-noise ratio (SNR).

        Arguments:
            snr: The signal-to-noise ratio (SNR) in dB. This can be $E_b/N_0$, $E_s/N_0$, $S/N$, $C/N_0$,
                or other SNR quantities. However, users are cautioned to be consistent for a given class instance.

        Returns:
            The number of errors at the specified SNR.

        Examples:
            See the class :ref:`error-rate-example` section.
        """
        return self._errors.get(snr, 0)

    def counts(self, snr: float) -> int:
        """
        Returns the number of counts at the specified signal-to-noise ratio (SNR).

        Arguments:
            snr: The signal-to-noise ratio (SNR) in dB. This can be $E_b/N_0$, $E_s/N_0$, $S/N$, $C/N_0$,
                or other SNR quantities. However, users are cautioned to be consistent for a given class instance.

        Returns:
            The number of counts at the specified SNR.

        Examples:
            See the class :ref:`error-rate-example` section.
        """
        return self._counts.get(snr, 0)

    def error_rate(self, snr: float | None) -> float:
        """
        Returns the error rate at the specified signal-to-noise ratio (SNR).

        Arguments:
            snr: The signal-to-noise ratio (SNR) in dB. This can be $E_b/N_0$, $E_s/N_0$, $S/N$, $C/N_0$,
                or other SNR quantities. However, users are cautioned to be consistent for a given class instance.

        Returns:
            The error rate at the specified SNR.

        Examples:
            See the class :ref:`error-rate-example` section.
        """
        return self.errors(snr) / self.counts(snr)

    def error_rates(self) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Returns all signal-to-noise ratios (SNRs) in ascending order and their corresponding error rates.

        Returns:
            - The signal-to-noise ratios (SNRs). The specific SNR quantity (e.g., $E_b/N_0$, $E_s/N_0$, $S/N$,
              $C/N_0$) is whatever was consistently provided to :meth:`add`.
            - The bit or symbol error rates.

        Examples:
            See the class :ref:`error-rate-example` section.
        """
        snrs = np.array(sorted(self._errors.keys()))
        error_rates = np.array([self.error_rate(snr) for snr in snrs])
        return snrs, error_rates
