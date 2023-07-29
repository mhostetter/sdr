import numpy as np

import sdr


def test_1():
    """
    Matlab:
        >> convertSNR(0:10, 'esno', 'ebno', 'BitsPerSymbol', 2, 'CodingRate', 1/3)'
    """
    esn0 = np.arange(0, 11)
    ebn0 = sdr.esn0_to_ebn0(esn0, 2, rate=1 / 3)
    ebn0_truth = np.array(
        [
            1.760912590556813,
            2.760912590556813,
            3.760912590556813,
            4.760912590556813,
            5.760912590556813,
            6.760912590556813,
            7.760912590556813,
            8.760912590556813,
            9.760912590556813,
            10.760912590556813,
            11.760912590556813,
        ]
    )
    assert np.allclose(ebn0, ebn0_truth)


def test_2():
    """
    Matlab:
        >> convertSNR(0:10, 'esno', 'ebno', 'BitsPerSymbol', 3, 'CodingRate', 1/2)'
    """
    esn0 = np.arange(0, 11)
    ebn0 = sdr.esn0_to_ebn0(esn0, 3, rate=1 / 2)
    ebn0_truth = np.array(
        [
            -1.760912590556812,
            -0.760912590556812,
            0.239087409443188,
            1.239087409443188,
            2.239087409443187,
            3.239087409443187,
            4.239087409443187,
            5.239087409443187,
            6.239087409443187,
            7.239087409443187,
            8.239087409443188,
        ]
    )
    assert np.allclose(ebn0, ebn0_truth)


def test_3():
    """
    Matlab:
        >> convertSNR(0:10, 'esno', 'ebno', 'BitsPerSymbol', 4, 'CodingRate', 2/3)'
    """
    esn0 = np.arange(0, 11)
    ebn0 = sdr.esn0_to_ebn0(esn0, 4, rate=2 / 3)
    ebn0_truth = np.array(
        [
            -4.259687322722811,
            -3.259687322722811,
            -2.259687322722811,
            -1.259687322722811,
            -0.259687322722811,
            0.740312677277189,
            1.740312677277189,
            2.740312677277189,
            3.740312677277189,
            4.740312677277189,
            5.740312677277189,
        ]
    )
    assert np.allclose(ebn0, ebn0_truth)
