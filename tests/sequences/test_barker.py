"""
Matlab:
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
        sdr.barker(13.0)
    with pytest.raises(ValueError):
        # Barker code of length 6 does not exist
        sdr.barker(6)


def test_1():
    seq = sdr.barker(1)
    seq_truth = np.array([-1])
    assert np.array_equal(seq, seq_truth)

    code = sdr.barker(1, output="binary")
    assert np.array_equal(code, (1 - seq_truth) // 2)


def test_2():
    seq = sdr.barker(2)
    seq_truth = np.array([-1, 1])
    assert np.array_equal(seq, seq_truth)

    code = sdr.barker(2, output="binary")
    assert np.array_equal(code, (1 - seq_truth) // 2)


def test_3():
    seq = sdr.barker(3)
    seq_truth = np.array([-1, -1, 1])
    assert np.array_equal(seq, seq_truth)

    code = sdr.barker(3, output="binary")
    assert np.array_equal(code, (1 - seq_truth) // 2)


def test_4():
    seq = sdr.barker(4)
    seq_truth = np.array([-1, -1, 1, -1])
    assert np.array_equal(seq, seq_truth)

    code = sdr.barker(4, output="binary")
    assert np.array_equal(code, (1 - seq_truth) // 2)


def test_5():
    seq = sdr.barker(5)
    seq_truth = np.array([-1, -1, -1, 1, -1])
    assert np.array_equal(seq, seq_truth)

    code = sdr.barker(5, output="binary")
    assert np.array_equal(code, (1 - seq_truth) // 2)


def test_7():
    seq = sdr.barker(7)
    seq_truth = np.array([-1, -1, -1, 1, 1, -1, 1])
    assert np.array_equal(seq, seq_truth)

    code = sdr.barker(7, output="binary")
    assert np.array_equal(code, (1 - seq_truth) // 2)


def test_11():
    seq = sdr.barker(11)
    seq_truth = np.array([-1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1])
    assert np.array_equal(seq, seq_truth)

    code = sdr.barker(11, output="binary")
    assert np.array_equal(code, (1 - seq_truth) // 2)


def test_13():
    seq = sdr.barker(13)
    seq_truth = np.array([-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1])
    assert np.array_equal(seq, seq_truth)

    code = sdr.barker(13, output="binary")
    assert np.array_equal(code, (1 - seq_truth) // 2)
