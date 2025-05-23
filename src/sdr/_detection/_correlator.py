"""
A module containing correlation-based detectors.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .._conversion import linear
from .._helper import convert_output, export, verify_arraylike, verify_scalar
from .._probability import Q, Qinv


@export
class ReplicaCorrelator:
    r"""
    Implements an clairvoyant replica-correlator detector.

    Notes:
        The replica-correlator detector is a clairvoyant detector that assumes perfect knowledge of the signal
        $s[n]$. The complex noise $w[n] \sim \mathcal{CN}(0, \sigma^2)$. The null and alternative hypotheses are
        given by:

        $$\mathcal{H}_0: x[n] = w[n]$$
        $$\mathcal{H}_1: x[n] = s[n] + w[n]$$

        The test statistic $T(x)$ is given by:

        $$T(x) = \mathrm{Re}\left( \sum\limits_{n=0}^{N-1} x[n]s^*[n] \right) > \gamma'$$
        $$
        T(x) \sim
        \begin{cases}
        \mathcal{N}\left(0, \sigma^2 \mathcal{E} / 2 \right) & \text{under } \mathcal{H}_0 \\
        \mathcal{N}\left(\mathcal{E}, \sigma^2 \mathcal{E} / 2 \right) & \text{under } \mathcal{H}_1 \\
        \end{cases}
        $$

        where $\mathcal{E}$ is the received energy $\mathcal{E} = \sum\limits_{n=0}^{N-1} \left| s[n] \right|^2$.

        The probability of detection $P_d$, probability of false alarm $P_{fa}$, and detection threshold
        $\gamma'$ are given by:

        $$P_d = Q\left( Q^{-1}(P_{fa}) - \sqrt{\frac{2 \mathcal{E}}{\sigma^2}} \right)$$
        $$P_{fa} = Q\left(\frac{\gamma'}{\sqrt{\sigma^2 \mathcal{E} / 2}}\right)$$
        $$\gamma' = \sqrt{\sigma^2 \mathcal{E} / 2} Q^{-1}(P_{fa})$$

    References:
        - Steven Kay, *Fundamentals of Statistical Signal Processing: Detection Theory*, Sections 4.3.2 and 13.3.1.

    Group:
        detection-detectors
    """

    # def __init__(
    #     self,
    #     n_nc: int,
    #     p_fa: float = 1e-6,
    #     streaming: bool = False,
    # ):
    #     r"""
    #     Initializes the energy detector.

    #     Arguments:
    #         n_nc: The number of samples $N_{nc}$ to non-coherently integrate.
    #         p_fa: The desired probability of false alarm $P_{fa}$.
    #     """
    #     if not isinstance(n_nc, int):
    #         raise TypeError(f"Argument 'n_nc' must be an integer, not {type(n_nc)}.")
    #     if not n_nc >= 1:
    #         raise ValueError(f"Argument 'n_nc' must be greater than or equal to 1, not {n_nc}.")
    #     self._N_nc = n_nc

    #     if not isinstance(p_fa, float):
    #         raise TypeError(f"Argument 'p_fa' must be a float, not {type(p_fa)}.")
    #     if not 0 < p_fa < 1:
    #         raise ValueError(f"Argument 'p_fa' must be in the range (0, 1), not {p_fa}.")
    #     self._p_fa = p_fa

    #     if not isinstance(streaming, bool):
    #         raise TypeError(f"Argument 'streaming' must be a bool, not {type(streaming)}.")
    #     self._streaming = streaming

    #     self._fir = FIR(np.ones(n_nc), streaming=streaming)

    @staticmethod
    def roc(
        enr: float,
        p_fa: npt.ArrayLike | None = None,
        complex: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""
        Computes the receiver operating characteristic (ROC) curve.

        Arguments:
            enr: The received energy-to-noise ratio $\mathcal{E}/\sigma^2$ in dB.
            p_fa: The probability of false alarm $P_{fa}$. If `None`, the ROC curve is computed for
                `p_fa = np.logspace(-10, 0, 101)`.
            complex: Indicates whether the signal is complex.

        Returns:
            - The probability of false alarm $P_{fa}$.
            - The probability of detection $P_d$.

        Examples:
            .. ipython:: python

                @savefig sdr_ReplicaCorrelator_roc_1.svg
                plt.figure(); \
                sdr.plot.roc(*sdr.ReplicaCorrelator.roc(-10), label="ENR = -10 dB"); \
                sdr.plot.roc(*sdr.ReplicaCorrelator.roc(-5), label="ENR = -5 dB"); \
                sdr.plot.roc(*sdr.ReplicaCorrelator.roc(0), label="ENR = 0 dB"); \
                sdr.plot.roc(*sdr.ReplicaCorrelator.roc(5), label="ENR = 5 dB"); \
                sdr.plot.roc(*sdr.ReplicaCorrelator.roc(10), label="ENR = 10 dB"); \
                sdr.plot.roc(*sdr.ReplicaCorrelator.roc(15), label="ENR = 15 dB");
        """
        verify_scalar(enr, float=True)
        if p_fa is None:
            p_fa = np.logspace(-10, 0, 101)
        else:
            p_fa = verify_arraylike(p_fa, float=True, inclusive_min=0, inclusive_max=1)

        p_d = ReplicaCorrelator.p_d(enr, p_fa, complex=complex)

        return convert_output(p_fa), convert_output(p_d)

    @staticmethod
    def p_d(
        enr: npt.ArrayLike,
        p_fa: npt.ArrayLike,
        complex: bool = True,
    ) -> npt.NDArray[np.float64]:
        r"""
        Computes the probability of detection $P_d$.

        Arguments:
            enr: The received energy-to-noise ratio $\mathcal{E}/\sigma^2$ in dB.
            p_fa: The probability of false alarm $P_{fa}$.
            complex: Indicates whether the signal is complex.

        Returns:
            The probability of detection $P_d$.

        Notes:
            For real signals:

            $$P_d = Q\left( Q^{-1}(P_{fa}) - \sqrt{\frac{\mathcal{E}}{\sigma^2}} \right)$$

            For complex signals:

            $$P_d = Q\left( Q^{-1}(P_{fa}) - \sqrt{\frac{\mathcal{E}}{\sigma^2 / 2}} \right)$$

        References:
            - Steven Kay, *Fundamentals of Statistical Signal Processing: Detection Theory*,
              Equations 4.14 and 13.9.

        Examples:
            .. ipython:: python

                enr = np.linspace(0, 20, 101)

                @savefig sdr_ReplicaCorrelator_p_d_1.svg
                plt.figure(); \
                sdr.plot.p_d(enr, sdr.ReplicaCorrelator.p_d(enr, 1e-1), label="$P_{fa} = 10^{-1}$"); \
                sdr.plot.p_d(enr, sdr.ReplicaCorrelator.p_d(enr, 1e-2), label="$P_{fa} = 10^{-2}$"); \
                sdr.plot.p_d(enr, sdr.ReplicaCorrelator.p_d(enr, 1e-3), label="$P_{fa} = 10^{-3}$"); \
                sdr.plot.p_d(enr, sdr.ReplicaCorrelator.p_d(enr, 1e-4), label="$P_{fa} = 10^{-4}$"); \
                sdr.plot.p_d(enr, sdr.ReplicaCorrelator.p_d(enr, 1e-5), label="$P_{fa} = 10^{-5}$"); \
                sdr.plot.p_d(enr, sdr.ReplicaCorrelator.p_d(enr, 1e-6), label="$P_{fa} = 10^{-6}$"); \
                sdr.plot.p_d(enr, sdr.ReplicaCorrelator.p_d(enr, 1e-7), label="$P_{fa} = 10^{-7}$");
        """
        enr = verify_arraylike(enr, float=True)
        p_fa = verify_arraylike(p_fa, float=True, inclusive_min=0, inclusive_max=1)

        enr_linear = linear(enr)
        if not complex:
            d2 = enr_linear  # Deflection coefficient
        else:
            d2 = 2 * enr_linear  # Deflection coefficient
        p_d = Q(Qinv(p_fa) - np.sqrt(d2))

        return convert_output(p_d)

    @staticmethod
    def p_fa(
        threshold: npt.ArrayLike,
        energy: npt.ArrayLike,
        sigma2: npt.ArrayLike,
        complex: bool = True,
    ) -> npt.NDArray[np.float64]:
        r"""
        Computes the probability of false alarm $P_{fa}$.

        Arguments:
            threshold: The threshold $\gamma'$.
            energy: The received energy $\mathcal{E} = \sum_{i=0}^{N-1} \left| s[n] \right|^2$.
            sigma2: The noise variance $\sigma^2$.
            complex: Indicates whether the signal is complex.

        Returns:
            The probability of false alarm $P_{fa}$.

        Notes:
            For real signals:

            $$P_{fa} = Q\left( \frac{\gamma'}{\sqrt{\sigma^2 \mathcal{E}}} \right)$$

            For complex signals:

            $$P_{fa} = Q\left( \frac{\gamma'}{\sqrt{\sigma^2 \mathcal{E} / 2}} \right)$$

        References:
            - Steven Kay, *Fundamentals of Statistical Signal Processing: Detection Theory*,
              Equations 4.12 and 13.6.
        """
        threshold = verify_arraylike(threshold, float=True)
        energy = verify_arraylike(energy, float=True, non_negative=True)
        sigma2 = verify_arraylike(sigma2, float=True, non_negative=True)

        if not complex:
            p_fa = Q(threshold / np.sqrt(energy * sigma2))
        else:
            p_fa = Q(threshold / np.sqrt(energy * sigma2 / 2))

        return convert_output(p_fa)

    @staticmethod
    def threshold(
        p_fa: npt.ArrayLike,
        energy: npt.ArrayLike,
        sigma2: npt.ArrayLike,
        complex: bool = True,
    ) -> npt.NDArray[np.float64]:
        r"""
        Computes the threshold $\gamma'$.

        Arguments:
            p_fa: The probability of false alarm $P_{fa}$.
            energy: The received energy $\mathcal{E} = \sum_{i=0}^{N-1} \left| s[n] \right|^2$.
            sigma2: The noise variance $\sigma^2$.
            complex: Indicates whether the signal is complex.

        Returns:
            The threshold $\gamma'$.

        Notes:
            For real signals:

            $$\gamma' = \sqrt{\sigma^2 \mathcal{E}} Q^{-1}(P_{fa})$$

            For complex signals:

            $$\gamma' = \sqrt{\sigma^2 \mathcal{E} / 2} Q^{-1}(P_{fa})$$

        References:
            - Steven Kay, *Fundamentals of Statistical Signal Processing: Detection Theory*,
              Equations 4.12 and 13.6.
        """
        p_fa = verify_arraylike(p_fa, float=True, inclusive_min=0, inclusive_max=1)
        energy = verify_arraylike(energy, float=True, non_negative=True)
        sigma2 = verify_arraylike(sigma2, float=True, non_negative=True)

        # Compute the variance of the test statistic
        if not complex:
            gamma_prime = np.sqrt(energy * sigma2) * Qinv(p_fa)
        else:
            gamma_prime = np.sqrt(energy * sigma2 / 2) * Qinv(p_fa)

        return convert_output(gamma_prime)

    # def test_statistic(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
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

    # def detect(self, x: npt.ArrayLike, sigma2: float) -> npt.NDArray[np.float64]:
    #     r"""
    #     Detects the presence of a signal.

    #     Arguments:
    #         x: The received signal $x[n]$.
    #         sigma2: The noise variance $\sigma^2$.

    #     Returns:
    #         The decision statistic $d$.
    #     """
    #     T = self.test_statistic(x)
    #     gamma = self.threshold(self.n_nc, self.desired_p_fa, sigma2)
    #     return T >= gamma

    # @property
    # def n_nc(self) -> int:
    #     """
    #     The number of samples $N_{nc}$ to non-coherently integrate.
    #     """
    #     return self._N_nc

    # @property
    # def desired_p_fa(self) -> float:
    #     """
    #     The desired probability of false alarm $P_{fa}$.
    #     """
    #     return self._p_fa

    # @property
    # def streaming(self) -> bool:
    #     """
    #     Indicates whether the detector is in streaming mode.
    #     """
    #     return self._streaming
