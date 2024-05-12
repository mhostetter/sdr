import numpy as np
import pytest

import sdr


@pytest.mark.parametrize("n_nc", [1, 3, 71])
@pytest.mark.parametrize("n_c", [1, 7])
@pytest.mark.parametrize("complex", [True, False])
@pytest.mark.parametrize("detector", ["coherent", "linear", "square-law"])
def test_pdfs(detector, complex, n_c, n_nc):
    rng = np.random.default_rng()

    snr = rng.uniform(-10, 10)  # dB
    sigma2 = 1
    A2 = sdr.linear(snr) * sigma2

    if detector == "coherent":
        # Non-coherent integration is not possible when using a coherent detector
        n_nc = None

    # Compute the theoretical distributions
    h0 = sdr.h0(sigma2, detector=detector, complex=complex, n_c=n_c, n_nc=n_nc)
    h1 = sdr.h1_theory(snr, sigma2, detector=detector, complex=complex, n_c=n_c, n_nc=n_nc)

    # Given a false rate, determine the detection threshold and theoretical probability of detection
    p_fa = 1e-1
    threshold = sdr.threshold(p_fa, sigma2, detector=detector, complex=complex, n_c=n_c, n_nc=n_nc)
    p_d = sdr.p_d(snr, p_fa, detector=detector, complex=complex, n_c=n_c, n_nc=n_nc)

    # Simulate the input signal x for the h0 and h1 hypotheses
    trials = 10_000 * n_c * (1 if n_nc is None else n_nc)
    if complex:
        x_h0 = rng.normal(size=trials, scale=np.sqrt(sigma2 / 2)) + 1j * rng.normal(
            size=trials, scale=np.sqrt(sigma2 / 2)
        )
    else:
        x_h0 = rng.normal(size=trials, scale=np.sqrt(sigma2))
    x_h1 = np.sqrt(A2) + x_h0

    # Perform coherent integration
    x_h0 = np.sum(x_h0.reshape((-1, n_c)), axis=-1)
    x_h1 = np.sum(x_h1.reshape((-1, n_c)), axis=-1)

    # Apply the detection scheme
    if detector == "coherent":
        z_h0 = np.real(x_h0)
        z_h1 = np.real(x_h1)
    elif detector == "linear":
        z_h0 = np.abs(x_h0)
        z_h1 = np.abs(x_h1)
    elif detector == "square-law":
        z_h0 = np.abs(x_h0) ** 2
        z_h1 = np.abs(x_h1) ** 2

    # Perform non-coherent integration
    if n_nc is not None and n_nc > 1:
        z_h0 = np.sum(z_h0.reshape((-1, n_nc)), axis=-1)
        z_h1 = np.sum(z_h1.reshape((-1, n_nc)), axis=-1)

    # Measure probability of false alarm and detection
    p_fa_meas = np.mean(z_h0 > threshold)
    p_d_meas = np.mean(z_h1 > threshold)

    try:
        if h0.mean() == 0:
            assert np.mean(z_h0) == pytest.approx(h0.mean(), abs=0.1)
        else:
            assert np.mean(z_h0) == pytest.approx(h0.mean(), rel=0.2)
        assert np.var(z_h0) == pytest.approx(h0.var(), rel=0.2)

        assert np.mean(z_h1) == pytest.approx(h1.mean(), rel=0.2)
        assert np.var(z_h1) == pytest.approx(h1.var(), rel=0.2)

        assert p_fa_meas == pytest.approx(p_fa, rel=0.2)
        assert p_d_meas == pytest.approx(p_d, rel=0.2)
    except AssertionError as e:
        # import matplotlib.pyplot as plt

        # plt.figure()
        # sdr.plot.detector_pdfs(h0, h1, threshold, p_h0=1e-6, p_h1=1e-3)
        # plt.gca().set_prop_cycle(None)
        # plt.hist(z_h0, bins=101, histtype="step", density=True, label="H0")
        # plt.hist(z_h1, bins=101, histtype="step", density=True, label="H1")
        # plt.legend()
        # plt.suptitle(detector)
        # plt.show()

        raise e
