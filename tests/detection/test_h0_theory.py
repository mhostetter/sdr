"""
These test vectors were manually verified through simulation.
"""

import numpy as np

import sdr


def test_real_coherent():
    sigma2 = 1
    T_x = np.linspace(-3, 3, 21)
    h0 = sdr.h0_theory(sigma2, detector="coherent", complex=False)
    pdf = h0.pdf(T_x)
    pdf_truth = np.array(
        [
            [
                0.00443185,
                0.01042093,
                0.02239453,
                0.0439836,
                0.07895016,
                0.1295176,
                0.19418605,
                0.26608525,
                0.3332246,
                0.38138782,
                0.39894228,
                0.38138782,
                0.3332246,
                0.26608525,
                0.19418605,
                0.1295176,
                0.07895016,
                0.0439836,
                0.02239453,
                0.01042093,
                0.00443185,
            ]
        ]
    )
    assert np.allclose(pdf, pdf_truth)


def test_complex_coherent():
    sigma2 = 1
    T_x = np.linspace(-3, 3, 21)
    h0 = sdr.h0_theory(sigma2, detector="coherent")
    pdf = h0.pdf(T_x)
    pdf_truth = np.array(
        [
            6.96265260e-05,
            3.84962380e-04,
            1.77782434e-03,
            6.85782500e-03,
            2.20958617e-02,
            5.94651446e-02,
            1.33672174e-01,
            2.50984287e-01,
            3.93621716e-01,
            5.15630455e-01,
            5.64189584e-01,
            5.15630455e-01,
            3.93621716e-01,
            2.50984287e-01,
            1.33672174e-01,
            5.94651446e-02,
            2.20958617e-02,
            6.85782500e-03,
            1.77782434e-03,
            3.84962380e-04,
            6.96265260e-05,
        ]
    )
    assert np.allclose(pdf, pdf_truth)


def test_real_linear():
    sigma2 = 1
    T_x = np.linspace(0, 5, 21)
    h0 = sdr.h0_theory(sigma2, detector="linear", complex=False)
    pdf = h0.pdf(T_x)
    pdf_truth = np.array(
        [
            [
                7.97884561e-01,
                7.73336234e-01,
                7.04130654e-01,
                6.02274864e-01,
                4.83941449e-01,
                3.65298171e-01,
                2.59035191e-01,
                1.72554638e-01,
                1.07981933e-01,
                6.34793037e-02,
                3.50566010e-02,
                1.81871250e-02,
                8.86369682e-03,
                4.05809611e-03,
                1.74536539e-03,
                7.05191365e-04,
                2.67660452e-04,
                9.54372731e-05,
                3.19674822e-05,
                1.00590146e-05,
                2.97343903e-06,
            ]
        ]
    )
    assert np.allclose(pdf, pdf_truth)


def test_complex_linear():
    sigma2 = 1
    T_x = np.linspace(0, 5, 21)
    h0 = sdr.h0_theory(sigma2, detector="linear")
    pdf = h0.pdf(T_x)
    pdf_truth = np.array(
        [
            0.00000000e00,
            4.69706531e-01,
            7.78800783e-01,
            8.54674237e-01,
            7.35758882e-01,
            5.24028468e-01,
            3.16197674e-01,
            1.63697178e-01,
            7.32625556e-02,
            2.84837194e-02,
            9.65227068e-03,
            2.85766075e-03,
            7.40458825e-04,
            1.68142651e-04,
            3.34958217e-05,
            5.85861706e-06,
            9.00281398e-07,
            1.21611556e-07,
            1.44470525e-08,
            1.50992146e-09,
            1.38879439e-10,
        ]
    )
    assert np.allclose(pdf, pdf_truth)


def test_real_square_law():
    sigma2 = 1
    T_x = np.linspace(0, 10, 21)
    h0 = sdr.h0_theory(sigma2, detector="square-law", complex=False)
    pdf = h0.pdf(T_x)
    pdf_truth = np.array(
        [
            [
                np.inf,
                0.43939129,
                0.24197072,
                0.15386632,
                0.10377687,
                0.07228896,
                0.05139344,
                0.03705618,
                0.02699548,
                0.01982171,
                0.01464498,
                0.01087474,
                0.0081087,
                0.00606731,
                0.00455334,
                0.0034259,
                0.00258337,
                0.00195186,
                0.00147728,
                0.00111982,
                0.00085004,
            ]
        ]
    )
    assert np.allclose(pdf, pdf_truth)


def test_complex_square_law():
    sigma2 = 1
    T_x = np.linspace(0, 10, 21)
    h0 = sdr.h0_theory(sigma2, detector="square-law")
    pdf = h0.pdf(T_x)
    pdf_truth = np.array(
        [
            1.00000000e00,
            6.06530660e-01,
            3.67879441e-01,
            2.23130160e-01,
            1.35335283e-01,
            8.20849986e-02,
            4.97870684e-02,
            3.01973834e-02,
            1.83156389e-02,
            1.11089965e-02,
            6.73794700e-03,
            4.08677144e-03,
            2.47875218e-03,
            1.50343919e-03,
            9.11881966e-04,
            5.53084370e-04,
            3.35462628e-04,
            2.03468369e-04,
            1.23409804e-04,
            7.48518299e-05,
            4.53999298e-05,
        ]
    )
    assert np.allclose(pdf, pdf_truth)
