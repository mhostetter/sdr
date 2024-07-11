import numpy as np

import sdr


def test_encode_1():
    """
    MATLAB:
        >> diff_enc = comm.DifferentialEncoder;
        >> x = randi([0 1], 10, 1); x
        >> diff_enc(x)
    """
    x = [0, 1, 0, 0, 1, 1, 0, 0, 0, 0]
    y = sdr.diff_encode(x)
    y_truth = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    assert np.array_equal(y, y_truth)


# TODO: Add more MATLAB tests
