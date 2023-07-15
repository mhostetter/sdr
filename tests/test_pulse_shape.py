import numpy as np
import pytest

import sdr


def test_root_raised_cosine_exceptions():
    with pytest.raises(ValueError):
        # Alpha must be non-negative
        sdr.root_raised_cosine(-0.1, 4, 6)
    with pytest.raises(ValueError):
        # Need at least 1 samples per symbol
        sdr.root_raised_cosine(0.5, 1, 6)
    with pytest.raises(ValueError):
        # Need at least 2 symbols
        sdr.root_raised_cosine(0.5, 4, 1)
    with pytest.raises(ValueError):
        # The filter must have even order
        sdr.root_raised_cosine(0.5, 3, 5)


def test_root_raised_cosine_0p1_6_4():
    """
    Matlab:
        >> h = rcosdesign(0.1, 6, 4); h'
    """
    h = sdr.root_raised_cosine(0.1, 4, 6)
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


def test_root_raised_cosine_0p5_6_4():
    """
    Matlab:
        >> h = rcosdesign(0.5, 6, 4); h'
    """
    h = sdr.root_raised_cosine(0.5, 4, 6)
    h_truth = np.array(
        [
            0.0015,
            -0.0082,
            -0.0075,
            0.0077,
            0.0212,
            0.0077,
            -0.0375,
            -0.0784,
            -0.0531,
            0.0784,
            0.2894,
            0.4873,
            0.5684,
            0.4873,
            0.2894,
            0.0784,
            -0.0531,
            -0.0784,
            -0.0375,
            0.0077,
            0.0212,
            0.0077,
            -0.0075,
            -0.0082,
            0.0015,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)


def test_root_raised_cosine_0p9_6_4():
    """
    Matlab:
        >> h = rcosdesign(0.9, 6, 4); h'
    """
    h = sdr.root_raised_cosine(0.9, 4, 6)
    h_truth = np.array(
        [
            -0.0033,
            0.0040,
            0.0045,
            -0.0065,
            -0.0100,
            0.0065,
            0.0164,
            -0.0139,
            -0.0497,
            0.0134,
            0.2304,
            0.4995,
            0.6230,
            0.4995,
            0.2304,
            0.0134,
            -0.0497,
            -0.0139,
            0.0164,
            0.0065,
            -0.0100,
            -0.0065,
            0.0045,
            0.0040,
            -0.0033,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)
