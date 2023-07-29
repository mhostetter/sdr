import numpy as np

import sdr


def test_1():
    """
    Matlab:
        >> convertSNR(0:10, 'ebno', 'snr', 'BitsPerSymbol', 2, 'CodingRate', 1/3, 'SamplesPerSymbol', 1)'
    """
    ebn0 = np.arange(0, 11)
    snr = sdr.ebn0_to_snr(ebn0, 2, rate=1 / 3, sps=1)
    snr_truth = np.array(
        [
            -1.760912590556813,
            -0.760912590556813,
            0.239087409443187,
            1.239087409443187,
            2.239087409443187,
            3.239087409443187,
            4.239087409443187,
            5.239087409443187,
            6.239087409443187,
            7.239087409443187,
            8.239087409443187,
        ]
    )
    assert np.allclose(snr, snr_truth)


def test_2():
    """
    Matlab:
        >> convertSNR(0:10, 'ebno', 'snr', 'BitsPerSymbol', 3, 'CodingRate', 1/2, 'SamplesPerSymbol', 2)'
    """
    ebn0 = np.arange(0, 11)
    snr = sdr.ebn0_to_snr(ebn0, 3, rate=1 / 2, sps=2)
    snr_truth = np.array(
        [
            -1.249387366082999,
            -0.249387366082999,
            0.750612633917001,
            1.750612633917001,
            2.750612633917001,
            3.750612633917001,
            4.750612633917001,
            5.750612633917001,
            6.750612633917001,
            7.750612633917001,
            8.750612633917001,
        ]
    )
    assert np.allclose(snr, snr_truth)


def test_3():
    """
    Matlab:
        >> convertSNR(0:10, 'ebno', 'snr', 'BitsPerSymbol', 4, 'CodingRate', 2/3, 'SamplesPerSymbol', 4)'
    """
    ebn0 = np.arange(0, 11)
    snr = sdr.ebn0_to_snr(ebn0, 4, rate=2 / 3, sps=4)
    snr_truth = np.array(
        [
            -1.760912590556813,
            -0.760912590556813,
            0.239087409443187,
            1.239087409443187,
            2.239087409443187,
            3.239087409443187,
            4.239087409443187,
            5.239087409443187,
            6.239087409443187,
            7.239087409443187,
            8.239087409443187,
        ]
    )
    assert np.allclose(snr, snr_truth)
