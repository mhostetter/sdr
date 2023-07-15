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
        >> h = rcosdesign(0.5, 6, 4, 'normal'); h'
    """
    h = sdr.raised_cosine(0.5, 6, 4)
    h_truth = np.array(
        [
            0.0000,
            0.0026,
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
            0.0026,
            0.0000,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)


def test_0p9_6_4():
    """
    Matlab:
        >> h = rcosdesign(0.9, 6, 4, 'normal'); h'
    """
    h = sdr.raised_cosine(0.9, 6, 4)
    h_truth = np.array(
        [
            0.0000,
            -0.0002,
            -0.0027,
            -0.0037,
            0.0000,
            0.0019,
            -0.0087,
            -0.0233,
            0.0000,
            0.1083,
            0.2977,
            0.4876,
            0.5680,
            0.4876,
            0.2977,
            0.1083,
            0.0000,
            -0.0233,
            -0.0087,
            0.0019,
            0.0000,
            -0.0037,
            -0.0027,
            -0.0002,
            0.0000,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)
