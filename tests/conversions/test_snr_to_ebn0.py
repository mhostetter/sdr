import numpy as np

import sdr


def test_1():
    """
    Matlab:
        >> convertSNR(0:10, 'snr', 'ebno', 'BitsPerSymbol', 2, 'CodingRate', 1/3, 'SamplesPerSymbol', 1)'
    """
    snr = np.arange(0, 11)
    ebn0 = sdr.snr_to_ebn0(snr, 2, rate=1 / 3, sps=1)
    ebn0_truth = np.array(
        [
            1.760912590556812,
            2.760912590556813,
            3.760912590556813,
            4.760912590556813,
            5.760912590556813,
            6.760912590556813,
            7.760912590556813,
            8.760912590556812,
            9.760912590556812,
            10.760912590556812,
            11.760912590556812,
        ]
    )
    assert np.allclose(ebn0, ebn0_truth)


def test_2():
    """
    Matlab:
        >> convertSNR(0:10, 'snr', 'ebno', 'BitsPerSymbol', 3, 'CodingRate', 1/2, 'SamplesPerSymbol', 2)'
    """
    snr = np.arange(0, 11)
    ebn0 = sdr.snr_to_ebn0(snr, 3, rate=1 / 2, sps=2)
    ebn0_truth = np.array(
        [
            1.249387366082999,
            2.249387366082999,
            3.249387366082999,
            4.249387366082999,
            5.249387366082999,
            6.249387366082999,
            7.249387366082999,
            8.249387366082999,
            9.249387366082999,
            10.249387366082999,
            11.249387366082999,
        ]
    )
    assert np.allclose(ebn0, ebn0_truth)


def test_3():
    """
    Matlab:
        >> convertSNR(0:10, 'snr', 'ebno', 'BitsPerSymbol', 4, 'CodingRate', 2/3, 'SamplesPerSymbol', 4)'
    """
    snr = np.arange(0, 11)
    ebn0 = sdr.snr_to_ebn0(snr, 4, rate=2 / 3, sps=4)
    ebn0_truth = np.array(
        [
            1.760912590556812,
            2.760912590556813,
            3.760912590556813,
            4.760912590556813,
            5.760912590556813,
            6.760912590556813,
            7.760912590556813,
            8.760912590556812,
            9.760912590556812,
            10.760912590556812,
            11.760912590556812,
        ]
    )
    assert np.allclose(ebn0, ebn0_truth)
