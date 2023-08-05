import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # x_c must be 1D
        x_c = np.zeros((2, 2), dtype=np.complex64)
        sdr.to_real_pb(x_c)
    with pytest.raises(ValueError):
        # x_c must be real
        x_c = np.zeros(4, dtype=np.float32)
        sdr.to_real_pb(x_c)
