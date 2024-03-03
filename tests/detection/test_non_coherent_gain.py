import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # Number of non-coherent samples must be at least 1
        sdr.non_coherent_gain(0, 10)
    with pytest.raises(ValueError):
        # P_FA must be between 0 and 1
        sdr.non_coherent_gain(1, 10, -0.1)
    with pytest.raises(ValueError):
        # P_FA must be between 0 and 1
        sdr.non_coherent_gain(1, 10, 1.1)


def test_scalar_snr_in():
    assert sdr.non_coherent_gain(1, 10, snr_ref="input") == pytest.approx(0.0)
    assert sdr.non_coherent_gain(2, 10, snr_ref="input") == pytest.approx(2.499445060713011)
    assert sdr.non_coherent_gain(10, 10, snr_ref="input") == pytest.approx(8.666092814306324)
    assert sdr.non_coherent_gain(20, 10, snr_ref="input") == pytest.approx(11.457842213037324)


def test_scalar_snr_out():
    assert sdr.non_coherent_gain(1, 10, snr_ref="output") == pytest.approx(0.0)
    assert sdr.non_coherent_gain(2, 10, snr_ref="output") == pytest.approx(2.3760378084124643)
    assert sdr.non_coherent_gain(10, 10, snr_ref="output") == pytest.approx(7.367080094838606)
    assert sdr.non_coherent_gain(20, 10, snr_ref="output") == pytest.approx(9.296533509663906)


def test_vector_snr_in():
    snr = 10
    n_nc = np.array([1, 2, 10, 20])
    g_nc = sdr.non_coherent_gain(n_nc, snr, snr_ref="input")
    g_nc_truth = np.array([0.0, 2.499445060713011, 8.666092814306324, 11.457842213037324])
    assert isinstance(g_nc, np.ndarray)
    assert np.allclose(g_nc, g_nc_truth)


def test_vector_snr_out():
    snr = 10
    n_nc = np.array([1, 2, 10, 20])
    g_nc = sdr.non_coherent_gain(n_nc, snr, snr_ref="output")
    g_nc_truth = np.array([0.0, 2.3760378084124643, 7.367080094838606, 9.296533509663906])
    assert isinstance(g_nc, np.ndarray)
    assert np.allclose(g_nc, g_nc_truth)
