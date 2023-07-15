import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # Alpha must be non-negative
        sdr.root_raised_cosine(-0.1, 6, 4)
    with pytest.raises(ValueError):
        # Need at least 1 samples per symbol
        sdr.root_raised_cosine(0.5, 6, 1)
    with pytest.raises(ValueError):
        # Need at least 2 symbols
        sdr.root_raised_cosine(0.5, 1, 4)
    with pytest.raises(ValueError):
        # The filter must have even order
        sdr.root_raised_cosine(0.5, 5, 3)


def test_0p1_6_4():
    """
    Matlab:
        >> h = rcosdesign(0.1, 6, 4); h'
    """
    h = sdr.root_raised_cosine(0.1, 6, 4)
    h_truth = np.array(
        [
            -0.0126,
            0.0277,
            0.0583,
            0.0559,
            0.0132,
            -0.0524,
            -0.1033,
            -0.0981,
            -0.0136,
            0.1401,
            0.3193,
            0.4625,
            0.5171,
            0.4625,
            0.3193,
            0.1401,
            -0.0136,
            -0.0981,
            -0.1033,
            -0.0524,
            0.0132,
            0.0559,
            0.0583,
            0.0277,
            -0.0126,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)


def test_0p5_6_4():
    """
    Matlab:
        >> h = rcosdesign(0.5, 5, 4); h'
    """
    h = sdr.root_raised_cosine(0.5, 5, 4)
    h_truth = np.array(
        [
            -0.0075,
            0.0077,
            0.0212,
            0.0077,
            -0.0375,
            -0.0784,
            -0.0531,
            0.0784,
            0.2894,
            0.4874,
            0.5685,
            0.4874,
            0.2894,
            0.0784,
            -0.0531,
            -0.0784,
            -0.0375,
            0.0077,
            0.0212,
            0.0077,
            -0.0075,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)


def test_0p9_6_4():
    """
    Matlab:
        >> h = rcosdesign(0.9, 6, 5); h'
    """
    h = sdr.root_raised_cosine(0.9, 6, 5)
    h_truth = np.array(
        [
            -0.0029,
            0.0023,
            0.0054,
            0.0008,
            -0.0077,
            -0.0090,
            0.0021,
            0.0145,
            0.0083,
            -0.0210,
            -0.0444,
            -0.0105,
            0.1144,
            0.3064,
            0.4845,
            0.5572,
            0.4845,
            0.3064,
            0.1144,
            -0.0105,
            -0.0444,
            -0.0210,
            0.0083,
            0.0145,
            0.0021,
            -0.0090,
            -0.0077,
            0.0008,
            0.0054,
            0.0023,
            -0.0029,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)
