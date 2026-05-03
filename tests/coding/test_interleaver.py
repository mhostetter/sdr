import numpy as np
import pytest

import sdr


def test_interleaver_map():
    interleaver = sdr.Interleaver([0, 3, 1, 2])

    np.testing.assert_array_equal(interleaver.map, [0, 3, 1, 2])
    np.testing.assert_array_equal(interleaver.inverse_map, [0, 2, 3, 1])
    assert len(interleaver) == 4


def test_interleaver_interleave():
    interleaver = sdr.Interleaver([0, 3, 1, 2])

    x = np.array([0, 1, 2, 3])
    y = interleaver.interleave(x)

    np.testing.assert_array_equal(y, [0, 2, 3, 1])


def test_interleaver_deinterleave():
    interleaver = sdr.Interleaver([0, 3, 1, 2])

    y = np.array([0, 2, 3, 1])
    x = interleaver.deinterleave(y)

    np.testing.assert_array_equal(x, [0, 1, 2, 3])


def test_interleaver_round_trip():
    interleaver = sdr.Interleaver([0, 3, 1, 2])

    x = np.arange(16)
    y = interleaver.interleave(x)
    z = interleaver.deinterleave(y)

    np.testing.assert_array_equal(z, x)


def test_interleaver_multiple_blocks():
    interleaver = sdr.Interleaver([0, 3, 1, 2])

    x = np.arange(8)
    y = interleaver.interleave(x)

    np.testing.assert_array_equal(y, [0, 2, 3, 1, 4, 6, 7, 5])
    np.testing.assert_array_equal(interleaver.deinterleave(y), x)


def test_interleaver_rejects_duplicate_map():
    with pytest.raises(ValueError):
        sdr.Interleaver([0, 1, 1, 2])


def test_interleaver_rejects_negative_map():
    with pytest.raises(ValueError):
        sdr.Interleaver([0, 1, 2, -1])


def test_interleaver_rejects_nonzero_based_map():
    with pytest.raises(ValueError):
        sdr.Interleaver([1, 2, 3, 4])


def test_interleaver_rejects_gap_map():
    with pytest.raises(ValueError):
        sdr.Interleaver([0, 1, 2, 4])


def test_interleaver_rejects_length_not_multiple():
    interleaver = sdr.Interleaver([0, 3, 1, 2])

    with pytest.raises(ValueError):
        interleaver.interleave(np.arange(5))

    with pytest.raises(ValueError):
        interleaver.deinterleave(np.arange(5))
