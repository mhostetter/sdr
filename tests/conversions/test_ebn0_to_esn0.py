import numpy as np

import sdr


def test_1():
    """
    Matlab:
        >> convertSNR(0:10, 'ebno', 'esno', 'BitsPerSymbol', 2, 'CodingRate', 1/3)'
    """
    ebn0 = np.arange(0, 11)
    esn0 = sdr.ebn0_to_esn0(ebn0, 2, rate=1 / 3)
    esn0_truth = np.array(
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
    assert np.allclose(esn0, esn0_truth)


def test_2():
    """
    Matlab:
        >> convertSNR(0:10, 'ebno', 'esno', 'BitsPerSymbol', 3, 'CodingRate', 1/2)'
    """
    ebn0 = np.arange(0, 11)
    esn0 = sdr.ebn0_to_esn0(ebn0, 3, rate=1 / 2)
    esn0_truth = np.array(
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
    assert np.allclose(esn0, esn0_truth)


def test_3():
    """
    Matlab:
        >> convertSNR(0:10, 'ebno', 'esno', 'BitsPerSymbol', 4, 'CodingRate', 2/3)'
    """
    ebn0 = np.arange(0, 11)
    esn0 = sdr.ebn0_to_esn0(ebn0, 4, rate=2 / 3)
    esn0_truth = np.array(
        [
            4.259687322722811,
            5.259687322722811,
            6.259687322722811,
            7.259687322722811,
            8.259687322722812,
            9.259687322722812,
            10.259687322722812,
            11.259687322722812,
            12.259687322722812,
            13.259687322722812,
            14.259687322722812,
        ]
    )
    assert np.allclose(esn0, esn0_truth)
