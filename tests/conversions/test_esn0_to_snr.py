import numpy as np

import sdr


def test_1():
    """
    Matlab:
        >> convertSNR(0:10, 'esno', 'snr', 'SamplesPerSymbol', 1)'
    """
    esn0 = np.arange(0, 11)
    snr = sdr.esn0_to_snr(esn0, sps=1)
    snr_truth = np.array(
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
    assert np.allclose(snr, snr_truth)


def test_2():
    """
    Matlab:
        >> convertSNR(0:10, 'esno', 'snr', 'SamplesPerSymbol', 2)'
    """
    esn0 = np.arange(0, 11)
    snr = sdr.esn0_to_snr(esn0, sps=2)
    snr_truth = np.array(
        [
            -3.010299956639812,
            -2.010299956639812,
            -1.010299956639812,
            -0.010299956639812,
            0.989700043360188,
            1.989700043360188,
            2.989700043360188,
            3.989700043360188,
            4.989700043360187,
            5.989700043360187,
            6.989700043360187,
        ]
    )
    assert np.allclose(snr, snr_truth)


def test_3():
    """
    Matlab:
        >> convertSNR(0:10, 'esno', 'snr', 'SamplesPerSymbol', 4)'
    """
    esn0 = np.arange(0, 11)
    snr = sdr.esn0_to_snr(esn0, sps=4)
    snr_truth = np.array(
        [
            -6.020599913279624,
            -5.020599913279624,
            -4.020599913279624,
            -3.020599913279624,
            -2.020599913279624,
            -1.020599913279624,
            -0.020599913279624,
            0.979400086720376,
            1.979400086720376,
            2.979400086720376,
            3.979400086720376,
        ]
    )
    assert np.allclose(snr, snr_truth)
