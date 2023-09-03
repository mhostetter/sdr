"""
A module containing a class that implements an energy detector.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.stats

from .._conversion import linear
from .._helper import export


@export
class EnergyDetector:
    r"""
    Implements an energy detector.

    Notes:
        The null and alternative hypotheses are given by the following. The signal $s[n]$ is assumed to be
        a complex-valued random process with noise variance $\sigma_s^2$. The noise $w[n]$ is assumed to be
        a complex-valued random process with noise variance $\sigma^2$.

        $$\mathcal{H}_0: x[n] = w[n]$$
        $$\mathcal{H}_1: x[n] = s[n] + w[n]$$

        The test statistic $T(x)$ is given by:

        $$T(x) = \sum\limits_{n=0}^{N-1} \left| x[n] \right|^2 > \gamma'$$
        $$
        \frac{T(x)}{\sigma^2 / 2} \sim \chi_{2N}^2 & \text{ under } \mathcal{H}_0 \\
        \frac{T(x)}{(\sigma_s^2 + \sigma^2) / 2} \sim \chi_{2N}^2 & \text{ under } \mathcal{H}_1
        $$

        The probability of detection $P_D$, probability of false alarm $P_{FA}$, and detection threshold
        $\gamma'$ are given by:

        $$P_D = Q_{\chi_{2N}^2}\left( \frac{Q_{\chi_{2N}^2}^{-1}(P_{FA})}{\sigma_s^2 /\sigma^2 + 1} \right)$$
        $$P_{FA} = Q_{\chi_{2N}^2}\left( \frac{\gamma'}{\sigma^2 / 2} \right)$$
        $$\gamma' = \frac{\sigma^2}{2} Q_{\chi_N^2}^{-1}(P_{FA})$$

    References:
        - Steven Kay, *Fundamentals of Statistical Signal Processing: Detection Theory*, Sections 5.3.

    Group:
        detection-detectors
    """

    # def __init__(
    #     self,
    #     N_nc: int,
    #     p_fa: float = 1e-6,
    #     streaming: bool = False,
    # ):
    #     r"""
    #     Initializes the energy detector.

    #     Arguments:
    #         N_nc: The number of samples $N_{NC}$ to non-coherently integrate.
    #         p_fa: The desired probability of false alarm $P_{FA}$.
    #     """
    #     if not isinstance(N_nc, int):
    #         raise TypeError(f"Argument 'N_nc' must be an integer, not {type(N_nc)}.")
    #     if not N_nc >= 1:
    #         raise ValueError(f"Argument 'N_nc' must be greater than or equal to 1, not {N_nc}.")
    #     self._N_nc = N_nc

    #     if not isinstance(p_fa, float):
    #         raise TypeError(f"Argument 'p_fa' must be a float, not {type(p_fa)}.")
    #     if not 0 < p_fa < 1:
    #         raise ValueError(f"Argument 'p_fa' must be in the range (0, 1), not {p_fa}.")
    #     self._p_fa = p_fa

    #     if not isinstance(streaming, bool):
    #         raise TypeError(f"Argument 'streaming' must be a bool, not {type(streaming)}.")
    #     self._streaming = streaming

    #     self._fir = FIR(np.ones(N_nc), streaming=streaming)

    @staticmethod
    def roc(
        snr: float,
        N_nc: float,
        p_fa: npt.ArrayLike | None = None,
        complex: bool = True,  # pylint: disable=redefined-builtin
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Computes the receiver operating characteristic (ROC) curve.

        Arguments:
            snr: The received signal-to-noise ratio $\sigma_s^2 / \sigma^2$ in dB.
            N_nc: The number of samples $N_{NC}$ to non-coherently integrate.
            p_fa: The probability of false alarm $P_{FA}$. If `None`, the ROC curve is computed for
                `p_fa = np.logspace(-10, 0, 101)`.
            complex: Indicates whether the signal is complex.

        Returns:
            - The probability of false alarm $P_{FA}$.
            - The probability of detection $P_D$.

        Examples:
            Plot the theoretical ROC curves for integrating a single sample at various SNRs.

            .. ipython:: python

                @savefig sdr_EnergyDetector_roc_1.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.roc(*sdr.EnergyDetector.roc(-20, 1), label=f"SNR = -20 dB"); \
                sdr.plot.roc(*sdr.EnergyDetector.roc(-10, 1), label=f"SNR = -10 dB"); \
                sdr.plot.roc(*sdr.EnergyDetector.roc(0, 1), label=f"SNR = -0 dB"); \
                sdr.plot.roc(*sdr.EnergyDetector.roc(10, 1), label=f"SNR = 10 dB"); \
                sdr.plot.roc(*sdr.EnergyDetector.roc(20, 1), label=f"SNR = 20 dB");

            Plot the theoretical ROC curves for various integration lengths at -10 dB SNR.

            .. ipython:: python

                @savefig sdr_EnergyDetector_roc_2.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.roc(*sdr.EnergyDetector.roc(-10, 1), label=f"N = 1"); \
                sdr.plot.roc(*sdr.EnergyDetector.roc(-10, 10), label=f"N = 10"); \
                sdr.plot.roc(*sdr.EnergyDetector.roc(-10, 100), label=f"N = 100"); \
                sdr.plot.roc(*sdr.EnergyDetector.roc(-10, 1_000), label=f"N = 1,000"); \
                sdr.plot.roc(*sdr.EnergyDetector.roc(-10, 5_000), label=f"N = 5,000");
        """
        if not isinstance(snr, (int, float)):
            raise TypeError(f"Argument 'snr' must be a number, not {type(snr)}.")
        if not isinstance(N_nc, int):
            raise TypeError(f"Argument 'N_nc' must be an integer, not {type(N_nc)}.")

        if p_fa is None:
            p_fa = np.logspace(-10, 0, 101)
        else:
            p_fa = np.asarray(p_fa)

        p_d = EnergyDetector.p_d(snr, N_nc, p_fa, complex=complex)

        return p_fa, p_d

    @staticmethod
    def p_d(
        snr: npt.ArrayLike,
        N_nc: npt.ArrayLike,
        p_fa: npt.ArrayLike,
        complex: bool = True,  # pylint: disable=redefined-builtin
    ) -> np.ndarray:
        r"""
        Computes the probability of detection $P_D$.

        Arguments:
            snr: The received signal-to-noise ratio $\sigma_s^2 / \sigma^2$ in dB.
            N_nc: The number of samples $N_{NC}$ to non-coherently integrate.
            p_fa: The probability of false alarm $P_{FA}$.

        Returns:
            The probability of detection $P_D$.

        Notes:
            For real signals:

            $$
            P_D &= Q_{\chi_N^2}\left( \frac{\sigma^2 Q_{\chi_N^2}^{-1}(P_{FA})}{\sigma_s^2 + \sigma^2} \right) \\
            &= Q_{\chi_N^2}\left( \frac{Q_{\chi_N^2}^{-1}(P_{FA})}{\sigma_s^2 /\sigma^2 + 1} \right)
            $$

            For complex signals:

            $$P_D = Q_{\chi_{2N}^2}\left( \frac{Q_{\chi_{2N}^2}^{-1}(P_{FA})}{\sigma_s^2 /\sigma^2 + 1} \right)$$

        References:
            - Steven Kay, *Fundamentals of Statistical Signal Processing: Detection Theory*,
              Equation 5.3.

        Examples:
            .. ipython:: python

                snr = np.linspace(-20, 10, 101)

                @savefig sdr_EnergyDetector_p_d_1.png
                plt.figure(figsize=(8, 4)); \
                sdr.plot.p_d(snr, sdr.EnergyDetector.p_d(snr, 25, 1e-1), label="$P_{FA} = 10^{-1}$"); \
                sdr.plot.p_d(snr, sdr.EnergyDetector.p_d(snr, 25, 1e-2), label="$P_{FA} = 10^{-2}$"); \
                sdr.plot.p_d(snr, sdr.EnergyDetector.p_d(snr, 25, 1e-3), label="$P_{FA} = 10^{-3}$"); \
                sdr.plot.p_d(snr, sdr.EnergyDetector.p_d(snr, 25, 1e-4), label="$P_{FA} = 10^{-4}$"); \
                sdr.plot.p_d(snr, sdr.EnergyDetector.p_d(snr, 25, 1e-5), label="$P_{FA} = 10^{-5}$");
        """
        snr = np.asarray(snr)
        N_nc = np.asarray(N_nc)
        p_fa = np.asarray(p_fa)

        snr_linear = linear(snr)

        if not complex:
            nu = N_nc  # Degrees of freedom
            gamma_dprime = scipy.stats.chi2.isf(p_fa, nu)  # Normalized threshold
            p_d = scipy.stats.chi2.sf(gamma_dprime / (snr_linear + 1), nu)
        else:
            nu = 2 * N_nc  # Degrees of freedom
            gamma_dprime = scipy.stats.chi2.isf(p_fa, nu)  # Normalized threshold
            p_d = scipy.stats.chi2.sf(gamma_dprime / (snr_linear + 1), nu)

        return p_d

    @staticmethod
    def p_fa(
        threshold: npt.ArrayLike,
        N_nc: npt.ArrayLike,
        sigma2: npt.ArrayLike,
        complex: bool = True,  # pylint: disable=redefined-builtin
    ) -> np.ndarray:
        r"""
        Computes the probability of false alarm $P_{FA}$.

        Arguments:
            threshold: The threshold $\gamma'$.
            N_nc: The number of samples $N_{NC}$ to non-coherently integrate.
            sigma2: The noise variance $\sigma^2$.
            complex: Indicates whether the signal is complex.

        Returns:
            The probability of false alarm $P_{FA}$.

        Notes:
            For real signals:

            $$P_{FA} = Q_{\chi_N^2}\left( \frac{\gamma'}{\sigma^2} \right)$$

            For complex signals:

            $$P_{FA} = Q_{\chi_{2N}^2}\left( \frac{\gamma'}{\sigma^2 / 2} \right)$$

        References:
            - Steven Kay, *Fundamentals of Statistical Signal Processing: Detection Theory*,
              Equation 5.2.
        """
        threshold = np.asarray(threshold)
        N_nc = np.asarray(N_nc)
        sigma2 = np.asarray(sigma2)

        if not complex:
            nu = N_nc  # Degrees of freedom
            p_fa = scipy.stats.chi2.sf(threshold / sigma2, nu)
        else:
            nu = 2 * N_nc  # Degrees of freedom
            p_fa = scipy.stats.chi2.sf(threshold / (sigma2 / 2), nu)

        return p_fa

    @staticmethod
    def threshold(
        N_nc: npt.ArrayLike,
        p_fa: npt.ArrayLike,
        sigma2: npt.ArrayLike,
        complex: bool = True,  # pylint: disable=redefined-builtin
    ) -> np.ndarray:
        r"""
        Computes the threshold $\gamma'$.

        Arguments:
            N_nc: The number of samples $N_{NC}$ to non-coherently integrate.
            p_fa: The probability of false alarm $P_{FA}$.
            sigma2: The noise variance $\sigma^2$.
            complex: Indicates whether the signal is complex.

        Returns:
            The threshold $\gamma'$.

        Notes:
            For real signals:

            $$\gamma' = \sigma^2 Q_{\chi_N^2}^{-1}(P_{FA})$$

            For complex signals:

            $$\gamma' = \frac{\sigma^2}{2} Q_{\chi_N^2}^{-1}(P_{FA})$$

        References:
            - Steven Kay, *Fundamentals of Statistical Signal Processing: Detection Theory*,
              Equation 5.2.
        """
        N_nc = np.asarray(N_nc)
        p_fa = np.asarray(p_fa)
        sigma2 = np.asarray(sigma2)

        if not complex:
            nu = N_nc  # Degrees of freedom
            gamma_prime = sigma2 * scipy.stats.chi2.isf(p_fa, nu)
        else:
            nu = 2 * N_nc  # Degrees of freedom
            gamma_prime = sigma2 / 2 * scipy.stats.chi2.isf(p_fa, nu)

        return gamma_prime

    # def test_statistic(self, x: npt.ArrayLike) -> np.ndarray:
    #     """
    #     Computes the test statistic $T(x)$.

    #     Arguments:
    #         x: The received signal $x[n]$.

    #     Returns:
    #         The test statistic $T(x)$.
    #     """
    #     x = np.asarray(x)

    #     if self.streaming:
    #         return self._fir(np.abs(x) ** 2)

    #     return self._fir(np.abs(x) ** 2, mode="same")

    # def detect(self, x: npt.ArrayLike, sigma2: float) -> np.ndarray:
    #     r"""
    #     Detects the presence of a signal.

    #     Arguments:
    #         x: The received signal $x[n]$.
    #         sigma2: The noise variance $\sigma^2$.

    #     Returns:
    #         The decision statistic $d$.
    #     """
    #     T = self.test_statistic(x)
    #     gamma = self.threshold(self.N_nc, self.desired_p_fa, sigma2)
    #     return T >= gamma

    # @property
    # def N_nc(self) -> int:
    #     """
    #     The number of samples $N_{NC}$ to non-coherently integrate.
    #     """
    #     return self._N_nc

    # @property
    # def desired_p_fa(self) -> float:
    #     """
    #     The desired probability of false alarm $P_{FA}$.
    #     """
    #     return self._p_fa

    # @property
    # def streaming(self) -> bool:
    #     """
    #     Indicates whether the detector is in streaming mode.
    #     """
    #     return self._streaming
