import pytest

import sdr


def test_accumulate():
    ber = sdr.ErrorRate()
    x = [1, 1, 1, 1, 1]  # Reference bit vector

    x_hat = [1, 0, 1, 1, 1]  # Received bit vector
    assert ber.add(10, x, x_hat) == (1, 5, 0.2)
    assert ber.errors(10) == 1
    assert ber.counts(10) == 5
    assert ber.error_rate(10) == 0.2

    x_hat = [1, 0, 1, 0, 1]  # Received bit vector
    assert ber.add(10, x, x_hat) == (2, 5, 0.4)
    assert ber.errors(10) == 3
    assert ber.counts(10) == 10
    assert ber.error_rate(10) == 0.3


def test_no_snr():
    ber = sdr.ErrorRate()

    assert ber.errors(10) == 0
    assert ber.counts(10) == 0
    with pytest.raises(ZeroDivisionError):
        assert ber.error_rate(10)
