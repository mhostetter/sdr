import numpy as np

import sdr


def test_1():
    """
    Matlab:
        >> convertSNR(0:10, 'snr', 'esno', 'SamplesPerSymbol', 1)'
    """
    snr = np.arange(0, 11)
    esn0 = sdr.snr_to_esn0(snr, sps=1)
    esn0_truth = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ]
    )
    assert np.allclose(esn0, esn0_truth)


def test_2():
    """
    Matlab:
        >> convertSNR(0:10, 'snr', 'esno', 'SamplesPerSymbol', 2)'
    """
    snr = np.arange(0, 11)
    esn0 = sdr.snr_to_esn0(snr, sps=2)
    esn0_truth = np.array(
        [
            3.010299956639812,
            4.010299956639813,
            5.010299956639813,
            6.010299956639813,
            7.010299956639813,
            8.010299956639813,
            9.010299956639813,
            10.010299956639813,
            11.010299956639813,
            12.010299956639813,
            13.010299956639813,
        ]
    )
    assert np.allclose(esn0, esn0_truth)


def test_3():
    """
    Matlab:
        >> convertSNR(0:10, 'snr', 'esno', 'SamplesPerSymbol', 4)'
    """
    snr = np.arange(0, 11)
    esn0 = sdr.snr_to_esn0(snr, sps=4)
    esn0_truth = np.array(
        [
            6.020599913279624,
            7.020599913279624,
            8.020599913279625,
            9.020599913279625,
            10.020599913279625,
            11.020599913279625,
            12.020599913279625,
            13.020599913279625,
            14.020599913279625,
            15.020599913279625,
            16.020599913279625,
        ]
    )
    assert np.allclose(esn0, esn0_truth)
