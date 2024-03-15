"""
These test vectors were manually verified through simulation.
"""

import numpy as np

import sdr


def test_real_coherent():
    snr = np.arange(-10, 11)
    p_fa = 1e-3
    p_d = sdr.p_d(snr, p_fa, detector="coherent", complex=False)
    p_d_truth = np.array(
        [
            [
                0.00276855,
                0.00311505,
                0.00354992,
                0.0041021,
                0.00481212,
                0.0057375,
                0.0069609,
                0.00860275,
                0.01084068,
                0.01393962,
                0.01829847,
                0.02452172,
                0.03352737,
                0.04670335,
                0.06611846,
                0.09476801,
                0.13676489,
                0.19724273,
                0.28151531,
                0.39286892,
                0.52871709,
            ]
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_complex_coherent():
    snr = np.arange(-10, 11)
    p_fa = 1e-3
    p_d = sdr.p_d(snr, p_fa, detector="coherent")
    p_d_truth = np.array(
        [
            0.00410852,
            0.00482044,
            0.00574841,
            0.00697543,
            0.0086224,
            0.01086766,
            0.01397727,
            0.0183518,
            0.02459838,
            0.03363894,
            0.04686726,
            0.06636041,
            0.09512449,
            0.13728417,
            0.19798103,
            0.28252208,
            0.39415521,
            0.53020989,
            0.67799934,
            0.81475691,
            0.91649936,
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_real_linear():
    snr = np.arange(-10, 11)
    p_fa = 1e-3
    p_d = sdr.p_d(snr, p_fa, detector="linear", complex=False)
    p_d_truth = np.array(
        [
            [
                0.00162332,
                0.00179743,
                0.00202417,
                0.00232169,
                0.00271559,
                0.00324256,
                0.00395603,
                0.00493509,
                0.00629867,
                0.00822823,
                0.01100431,
                0.01506513,
                0.02109971,
                0.0301926,
                0.04403871,
                0.06523568,
                0.09761458,
                0.14644439,
                0.21809582,
                0.31841208,
                0.44897593,
            ]
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_complex_linear():
    snr = np.arange(-10, 11)
    p_fa = 1e-3
    p_d = sdr.p_d(snr, p_fa, detector="linear")
    p_d_truth = np.array(
        [
            0.00177786,
            0.0020086,
            0.00231693,
            0.00273403,
            0.00330629,
            0.00410404,
            0.00523576,
            0.00687179,
            0.00928377,
            0.01291049,
            0.01846703,
            0.02712174,
            0.0407706,
            0.06242657,
            0.09667487,
            0.14995286,
            0.23001292,
            0.34340965,
            0.48995306,
            0.65552537,
            0.81029237,
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_real_square_law():
    snr = np.arange(-10, 11)
    p_fa = 1e-3
    p_d = sdr.p_d(snr, p_fa, detector="square-law", complex=False)
    p_d_truth = np.array(
        [
            [
                0.00162332,
                0.00179743,
                0.00202417,
                0.00232169,
                0.00271559,
                0.00324256,
                0.00395603,
                0.00493509,
                0.00629867,
                0.00822823,
                0.01100431,
                0.01506513,
                0.02109971,
                0.0301926,
                0.04403871,
                0.06523568,
                0.09761458,
                0.14644439,
                0.21809582,
                0.31841208,
                0.44897593,
            ]
        ]
    )
    assert np.allclose(p_d, p_d_truth)


def test_complex_square_law():
    snr = np.arange(-10, 11)
    p_fa = 1e-3
    p_d = sdr.p_d(snr, p_fa, detector="square-law")
    p_d_truth = np.array(
        [
            0.00177786,
            0.0020086,
            0.00231693,
            0.00273403,
            0.00330629,
            0.00410404,
            0.00523576,
            0.00687179,
            0.00928377,
            0.01291049,
            0.01846703,
            0.02712174,
            0.0407706,
            0.06242657,
            0.09667487,
            0.14995286,
            0.23001292,
            0.34340965,
            0.48995306,
            0.65552537,
            0.81029237,
        ]
    )
    assert np.allclose(p_d, p_d_truth)
