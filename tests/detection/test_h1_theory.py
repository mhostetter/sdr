"""
These test vectors were manually verified through simulation.
"""
import numpy as np

import sdr


def test_real_coherent():
    snr = 5
    sigma2 = 1
    T_x = np.linspace(-1, 5, 21)
    h0 = sdr.h1_theory(snr, sigma2, detector="coherent", complex=False)
    pdf = h0.pdf(T_x)
    pdf_truth = np.array(
        [
            [
                0.00840981,
                0.01850206,
                0.03720213,
                0.06836425,
                0.11481634,
                0.17623487,
                0.24722564,
                0.31696307,
                0.37139608,
                0.39772186,
                0.38925581,
                0.34818034,
                0.28463407,
                0.21265863,
                0.1452087,
                0.09061827,
                0.05168356,
                0.02694031,
                0.01283412,
                0.00558783,
                0.00222348,
            ]
        ]
    )
    assert np.allclose(pdf, pdf_truth)


def test_complex_coherent():
    snr = 5
    sigma2 = 1
    T_x = np.linspace(-1, 5, 21)
    h0 = sdr.h1_theory(snr, sigma2, detector="coherent")
    pdf = h0.pdf(T_x)
    pdf_truth = np.array(
        [
            2.50713069e-04,
            1.21351548e-03,
            4.90614817e-03,
            1.65677294e-02,
            4.67317797e-02,
            1.10100335e-01,
            2.16666593e-01,
            3.56141234e-01,
            4.88967010e-01,
            5.60742989e-01,
            5.37124717e-01,
            4.29747558e-01,
            2.87196201e-01,
            1.60313818e-01,
            7.47463888e-02,
            2.91096154e-02,
            9.46912203e-03,
            2.57282325e-03,
            5.83898265e-04,
            1.10685673e-04,
            1.75255892e-05,
        ]
    )
    assert np.allclose(pdf, pdf_truth)


def test_real_linear():
    snr = 5
    sigma2 = 1
    T_x = np.linspace(0, 5, 21)
    h0 = sdr.h1_theory(snr, sigma2, detector="linear", complex=False)
    pdf = h0.pdf(T_x)
    pdf_truth = np.array(
        [
            [
                0.1641573,
                0.17509049,
                0.20600598,
                0.25145508,
                0.30310964,
                0.35105322,
                0.38564071,
                0.39957292,
                0.3895728,
                0.3570555,
                0.30751197,
                0.24882575,
                0.18914982,
                0.13507708,
                0.09061863,
                0.05710989,
                0.03381134,
                0.01880487,
                0.00982505,
                0.00482232,
                0.00222348,
            ]
        ]
    )
    assert np.allclose(pdf, pdf_truth)


def test_complex_linear():
    snr = 5
    sigma2 = 1
    T_x = np.linspace(0, 5, 21)
    h0 = sdr.h1_theory(snr, sigma2, detector="linear")
    pdf = h0.pdf(T_x)
    pdf_truth = np.array(
        [
            0.00000000e00,
            2.40103778e-02,
            6.46544495e-02,
            1.35478797e-01,
            2.41002988e-01,
            3.69678311e-01,
            4.92330033e-01,
            5.71722475e-01,
            5.80581147e-01,
            5.16600806e-01,
            4.03346287e-01,
            2.76619955e-01,
            1.66767408e-01,
            8.84347001e-02,
            4.12689720e-02,
            1.69542413e-02,
            6.13366797e-03,
            1.95460556e-03,
            5.48764932e-04,
            1.35762180e-04,
            2.96006854e-05,
        ]
    )
    assert np.allclose(pdf, pdf_truth)


def test_real_square_law():
    snr = 5
    sigma2 = 1
    T_x = np.linspace(0, 15, 21)
    h0 = sdr.h1_theory(snr, sigma2, detector="square-law", complex=False)
    pdf = h0.pdf(T_x)
    pdf_truth = np.array(
        [
            [
                0.0,
                0.15891004,
                0.14152641,
                0.1285469,
                0.11528469,
                0.10182914,
                0.08870543,
                0.07634447,
                0.06501986,
                0.05486974,
                0.04593176,
                0.03817495,
                0.03152497,
                0.02588304,
                0.02113936,
                0.01718237,
                0.01390464,
                0.01120652,
                0.00899801,
                0.0071995,
                0.00574169,
            ]
        ]
    )
    assert np.allclose(pdf, pdf_truth)


def test_complex_square_law():
    snr = 5
    sigma2 = 1
    T_x = np.linspace(0, 15, 21)
    h0 = sdr.h1_theory(snr, sigma2, detector="square-law")
    pdf = h0.pdf(T_x)
    pdf_truth = np.array(
        [
            0.0,
            0.1041543,
            0.14548005,
            0.16411001,
            0.164016,
            0.15121239,
            0.13144477,
            0.10919311,
            0.08746608,
            0.06798781,
            0.05152317,
            0.03820379,
            0.02779457,
            0.01988559,
            0.01401647,
            0.00974811,
            0.0066979,
            0.00455161,
            0.00306199,
            0.00204083,
            0.00134858,
        ]
    )
    assert np.allclose(pdf, pdf_truth)
