import numpy as np
import pytest

import sdr


@pytest.mark.parametrize("error", [1e-9, 1e-6, 1e-3, 1e-1])
def test_real_passband(error):
    sample_rate = 2e6
    freq = 100e3
    duration = 1000e-6
    x = sdr.sinusoid(duration, freq, sample_rate=sample_rate, complex=False)

    y = sdr.clock_error(x, error)

    error = -error / (1 + error)
    z = sdr.clock_error(y, error)

    # Ignore the first and last 5 sample due to filter effects. The signals aren't perfectly identical also due
    # to filter effects. Use abolute tolerance to verify the signals are spinning at different rates.
    np.testing.assert_allclose(x[5 : z.size - 5], z[5:-5], atol=0.01)


@pytest.mark.parametrize("error", [1e-9, 1e-6, 1e-3, 1e-1])
def test_complex_passband(error):
    sample_rate = 2e6
    center_freq = 1e6
    duration = 1000e-6
    x = sdr.sinusoid(duration, 0, sample_rate=sample_rate)

    y = sdr.clock_error(x, error, 0, center_freq, sample_rate=sample_rate)

    error = -error / (1 + error)
    z = sdr.clock_error(y, error, 0, center_freq, sample_rate=sample_rate)

    # Ignore the first and last 5 sample due to filter effects. The signals aren't perfectly identical also due
    # to filter effects. Use abolute tolerance to verify the signals are spinning at different rates.
    np.testing.assert_allclose(x[5 : z.size - 5], z[5:-5], atol=0.01)
