import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # x_r must be 1D
        x_r = np.zeros((2, 2), dtype=float)
        sdr.to_complex_bb(x_r)
    with pytest.raises(ValueError):
        # x_r must be real
        x_r = np.zeros(4, dtype=complex)
        sdr.to_complex_bb(x_r)
