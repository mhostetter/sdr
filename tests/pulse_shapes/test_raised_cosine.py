import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # Alpha must be non-negative
        sdr.raised_cosine(-0.1, 6, 4)
    with pytest.raises(ValueError):
        # Need at least 1 samples per symbol
        sdr.raised_cosine(0.5, 6, 1)
    with pytest.raises(ValueError):
        # Need at least 2 symbols
        sdr.raised_cosine(0.5, 1, 4)
    with pytest.raises(ValueError):
        # The filter must have even order
        sdr.raised_cosine(0.5, 5, 3)


def test_0p1_6_4():
    """
    Matlab:
        >> h = rcosdesign(0.1, 6, 4, 'normal'); h'
    """
    h = sdr.raised_cosine(0.1, 6, 4)
    h_truth = np.array(
        [
            0.0000,
            0.0389,
            0.0612,
            0.0487,
            -0.0000,
            -0.0637,
            -0.1060,
            -0.0905,
            0.0000,
            0.1523,
            0.3240,
            0.4590,
            0.5101,
            0.4590,
            0.3240,
            0.1523,
            0.0000,
            -0.0905,
            -0.1060,
            -0.0637,
            -0.0000,
            0.0487,
            0.0612,
            0.0389,
            0.0000,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)


def test_0p5_6_4():
    """
    Matlab:
        >> h = rcosdesign(0.5, 5, 4, 'normal'); h'
    """
    h = sdr.raised_cosine(0.5, 5, 4)
    h_truth = np.array(
        [
            0.0092,
            0.0122,
            -0.0000,
            -0.0308,
            -0.0642,
            -0.0655,
            0.0000,
            0.1403,
            0.3208,
            0.4743,
            0.5345,
            0.4743,
            0.3208,
            0.1403,
            0.0000,
            -0.0655,
            -0.0642,
            -0.0308,
            -0.0000,
            0.0122,
            0.0092,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)


def test_0p9_6_4():
    """
    Matlab:
        >> h = rcosdesign(0.9, 6, 5, 'normal'); h'
    """
    h = sdr.raised_cosine(0.9, 6, 5)
    h_truth = np.array(
        [
            0.0000,
            0.0001,
            -0.0014,
            -0.0032,
            -0.0029,
            0.0000,
            0.0020,
            -0.0025,
            -0.0141,
            -0.0209,
            0.0000,
            0.0705,
            0.1931,
            0.3399,
            0.4610,
            0.5080,
            0.4610,
            0.3399,
            0.1931,
            0.0705,
            0.0000,
            -0.0209,
            -0.0141,
            -0.0025,
            0.0020,
            0.0000,
            -0.0029,
            -0.0032,
            -0.0014,
            0.0001,
            0.0000,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)
