import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # Time-bandwidth product must be non-negative
        sdr.gaussian(-0.1, 4, 6)
    with pytest.raises(ValueError):
        # Need at least 2 symbols
        sdr.gaussian(0.5, 1, 4)
    with pytest.raises(ValueError):
        # Need at least 1 samples per symbol
        sdr.gaussian(0.5, 6, 1)
    with pytest.raises(ValueError):
        # The filter must have even order
        sdr.gaussian(0.5, 3, 5)


def test_0p1_4_8():
    """
    Matlab:
        >> h = gaussdesign(0.1, 4, 8); h'
    """
    h = sdr.gaussian(0.1, 4, 8)
    h_truth = np.array(
        [
            0.0137,
            0.0157,
            0.0179,
            0.0201,
            0.0225,
            0.0249,
            0.0274,
            0.0298,
            0.0321,
            0.0344,
            0.0364,
            0.0382,
            0.0398,
            0.0411,
            0.0420,
            0.0425,
            0.0427,
            0.0425,
            0.0420,
            0.0411,
            0.0398,
            0.0382,
            0.0364,
            0.0344,
            0.0321,
            0.0298,
            0.0274,
            0.0249,
            0.0225,
            0.0201,
            0.0179,
            0.0157,
            0.0137,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)


def test_0p2_5_8():
    """
    Matlab:
        >> h = gaussdesign(0.2, 5, 8); h'
    """
    h = sdr.gaussian(0.2, 5, 8)
    h_truth = np.array(
        [
            0.0001,
            0.0001,
            0.0002,
            0.0004,
            0.0008,
            0.0014,
            0.0023,
            0.0037,
            0.0058,
            0.0087,
            0.0127,
            0.0178,
            0.0241,
            0.0315,
            0.0397,
            0.0482,
            0.0566,
            0.0641,
            0.0701,
            0.0739,
            0.0753,
            0.0739,
            0.0701,
            0.0641,
            0.0566,
            0.0482,
            0.0397,
            0.0315,
            0.0241,
            0.0178,
            0.0127,
            0.0087,
            0.0058,
            0.0037,
            0.0023,
            0.0014,
            0.0008,
            0.0004,
            0.0002,
            0.0001,
            0.0001,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)


def test_0p3_4_9():
    """
    Matlab:
        >> h = gaussdesign(0.3, 4, 9); h'
    """
    h = sdr.gaussian(0.3, 4, 9)
    h_truth = np.array(
        [
            0.0000,
            0.0000,
            0.0000,
            0.0001,
            0.0002,
            0.0005,
            0.0011,
            0.0022,
            0.0042,
            0.0077,
            0.0132,
            0.0213,
            0.0321,
            0.0455,
            0.0605,
            0.0755,
            0.0884,
            0.0972,
            0.1004,
            0.0972,
            0.0884,
            0.0755,
            0.0605,
            0.0455,
            0.0321,
            0.0213,
            0.0132,
            0.0077,
            0.0042,
            0.0022,
            0.0011,
            0.0005,
            0.0002,
            0.0001,
            0.0000,
            0.0000,
            0.0000,
        ]
    )
    np.testing.assert_almost_equal(h, h_truth, decimal=3)
