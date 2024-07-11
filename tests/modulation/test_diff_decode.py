import numpy as np

import sdr


def test_decode_1():
    """
    MATLAB:
        >> diff_dec = comm.DifferentialDecoder;
        >> y = randi([0 1], 10, 1); y
        >> diff_dec(y)
    """
    y = [1, 0, 1, 1, 0, 0, 0, 0, 0, 1]
    x = sdr.diff_decode(y)
    x_truth = [1, 1, 1, 0, 1, 0, 0, 0, 0, 1]
    assert np.array_equal(x, x_truth)


# TODO: Add more MATLAB tests
