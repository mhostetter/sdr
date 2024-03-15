"""
MATLAB:
    for N = [1, 2, 3, 4, 5, 7, 11, 13]
        barker = comm.BarkerCode("Length", N, "SamplesPerFrame", N);
        disp(N)
        disp(out)
        disp(barker())
    end
"""

import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(TypeError):
        sdr.barker_code(13.0)
    with pytest.raises(ValueError):
        # Barker code of length 6 does not exist
        sdr.barker_code(6)

    with pytest.raises(ValueError):
        sdr.barker_code(13, output="invalid")


def verify_code(length, sequence_truth):
    sequence_truth = np.array(sequence_truth)

    sequence = sdr.barker_code(length, output="bipolar")
    assert isinstance(sequence, np.ndarray)
    assert np.array_equal(sequence, sequence_truth)

    code = sdr.barker_code(length, output="binary")
    assert isinstance(code, np.ndarray)
    assert np.array_equal(code, (1 - sequence_truth) // 2)


def test_1():
    verify_code(1, [-1])


def test_2():
    verify_code(2, [-1, 1])


def test_3():
    verify_code(3, [-1, -1, 1])


def test_4():
    verify_code(4, [-1, -1, 1, -1])


def test_5():
    verify_code(5, [-1, -1, -1, 1, -1])


def test_7():
    verify_code(7, [-1, -1, -1, 1, 1, -1, 1])


def test_11():
    verify_code(11, [-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1])


def test_13():
    verify_code(13, [-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1])
