import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # Erasure probability must be between 0 and 1
        sdr.bec_capacity(-0.1)
    with pytest.raises(ValueError):
        # Erasure probability must be between 0 and 1
        sdr.bec_capacity(1.1)


def test_outputs():
    C = sdr.bec_capacity([0, 0.5, 1])
    assert isinstance(C, np.ndarray)
    assert np.array_equal(C, [1, 0.5, 0])

    C = sdr.bec_capacity(0.5)
    assert isinstance(C, float)
    assert C == 0.5
