import numpy as np
import pytest

import sdr


def test_block_interleaver_map():
    interleaver = sdr.BlockInterleaver(3, 4)

    np.testing.assert_array_equal(interleaver.map, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])
    np.testing.assert_array_equal(interleaver.inverse_map, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
    assert len(interleaver) == 12


def test_block_interleaver_interleave():
    interleaver = sdr.BlockInterleaver(3, 4)

    x = np.arange(12)
    y = interleaver.interleave(x)

    np.testing.assert_array_equal(y, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])


def test_block_interleaver_deinterleave():
    interleaver = sdr.BlockInterleaver(3, 4)

    y = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
    x = interleaver.deinterleave(y)

    np.testing.assert_array_equal(x, np.arange(12))


def test_block_interleaver_round_trip():
    interleaver = sdr.BlockInterleaver(3, 4)

    x = np.arange(24)
    y = interleaver.interleave(x)
    z = interleaver.deinterleave(y)

    np.testing.assert_array_equal(z, x)


def test_block_interleaver_multiple_blocks():
    interleaver = sdr.BlockInterleaver(2, 3)

    x = np.arange(12)
    y = interleaver.interleave(x)

    np.testing.assert_array_equal(y, [0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11])
    np.testing.assert_array_equal(interleaver.deinterleave(y), x)


def test_block_interleaver_rejects_invalid_rows():
    with pytest.raises(ValueError):
        sdr.BlockInterleaver(0, 4)

    with pytest.raises(ValueError):
        sdr.BlockInterleaver(-1, 4)


def test_block_interleaver_rejects_invalid_cols():
    with pytest.raises(ValueError):
        sdr.BlockInterleaver(3, 0)

    with pytest.raises(ValueError):
        sdr.BlockInterleaver(3, -1)


def test_block_interleaver_rejects_length_not_multiple():
    interleaver = sdr.BlockInterleaver(3, 4)

    with pytest.raises(ValueError):
        interleaver.interleave(np.arange(13))

    with pytest.raises(ValueError):
        interleaver.deinterleave(np.arange(13))
