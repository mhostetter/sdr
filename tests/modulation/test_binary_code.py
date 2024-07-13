import numpy as np

import sdr


def test_1():
    code = sdr.binary_code(2)
    code_truth = [0, 1]
    assert isinstance(code, np.ndarray)
    assert np.array_equal(code, code_truth)


def test_2():
    code = sdr.binary_code(4)
    code_truth = [0, 1, 2, 3]
    assert isinstance(code, np.ndarray)
    assert np.array_equal(code, code_truth)


def test_3():
    code = sdr.binary_code(8)
    code_truth = [0, 1, 2, 3, 4, 5, 6, 7]
    assert isinstance(code, np.ndarray)
    assert np.array_equal(code, code_truth)


def test_4():
    code = sdr.binary_code(16)
    code_truth = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert isinstance(code, np.ndarray)
    assert np.array_equal(code, code_truth)
