import numpy as np
import pytest

import sdr


def test_absolute_limit():
    esn0 = -100  # dB
    C = sdr.awgn_capacity(esn0)
    ebn0 = esn0 - 10 * np.log10(C)
    ebn0_limit = 10 * np.log10(np.log(2))  # ~ -1.59 dB
    assert ebn0 == pytest.approx(ebn0_limit)
