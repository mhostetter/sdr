"""
These test vectors were manually verified through simulation.
"""

import numpy as np

import sdr


def test_real_coherent():
    threshold = np.arange(0, 21)
    sigma2 = 1
    p_fa = sdr.p_fa(threshold, sigma2, detector="coherent", complex=False)
    p_fa_truth = np.array(
        [
            [
                5.00000000e-01,
                1.58655254e-01,
                2.27501319e-02,
                1.34989803e-03,
                3.16712418e-05,
                2.86651572e-07,
                9.86587645e-10,
                1.27981254e-12,
                6.22096057e-16,
                1.12858841e-19,
                7.61985302e-24,
                1.91065957e-28,
                1.77648211e-33,
                6.11716440e-39,
                7.79353682e-45,
                3.67096620e-51,
                6.38875440e-58,
                4.10599620e-65,
                9.74094892e-73,
                8.52722395e-81,
                2.75362412e-89,
            ]
        ]
    )
    assert np.allclose(p_fa, p_fa_truth)


def test_complex_coherent():
    threshold = np.arange(0, 21)
    sigma2 = 1
    p_fa = sdr.p_fa(threshold, sigma2, detector="coherent")
    p_fa_truth = np.array(
        [
            5.00000000e-001,
            7.86496035e-002,
            2.33886749e-003,
            1.10452485e-005,
            7.70862895e-009,
            7.68729897e-013,
            1.07598684e-017,
            2.09191280e-023,
            5.61214859e-030,
            2.06851587e-037,
            1.04424379e-045,
            7.20433069e-055,
            6.78130585e-065,
            8.69778658e-076,
            1.51861492e-087,
            3.60649709e-100,
            1.16424288e-113,
            5.10614008e-128,
            3.04118462e-143,
            2.45886142e-159,
            2.69793281e-176,
        ]
    )
    assert np.allclose(p_fa, p_fa_truth)


def test_real_linear():
    threshold = np.arange(0, 21)
    sigma2 = 1
    p_fa = sdr.p_fa(threshold, sigma2, detector="linear", complex=False)
    p_fa_truth = np.array(
        [
            [
                1.00000000e00,
                3.17310508e-01,
                4.55002639e-02,
                2.69979606e-03,
                6.33424837e-05,
                5.73303144e-07,
                1.97317529e-09,
                2.55962509e-12,
                1.24419211e-15,
                2.25717681e-19,
                1.52397060e-23,
                3.82131915e-28,
                3.55296422e-33,
                1.22343288e-38,
                1.55870736e-44,
                7.34193240e-51,
                1.27775088e-57,
                8.21199240e-65,
                1.94818978e-72,
                1.70544479e-80,
                5.50724824e-89,
            ]
        ]
    )
    assert np.allclose(p_fa, p_fa_truth)


def test_complex_linear():
    threshold = np.arange(0, 21)
    sigma2 = 1
    p_fa = sdr.p_fa(threshold, sigma2, detector="linear")
    p_fa_truth = np.array(
        [
            1.00000000e000,
            3.67879441e-001,
            1.83156389e-002,
            1.23409804e-004,
            1.12535175e-007,
            1.38879439e-011,
            2.31952283e-016,
            5.24288566e-022,
            1.60381089e-028,
            6.63967720e-036,
            3.72007598e-044,
            2.82077009e-053,
            2.89464031e-063,
            4.02006022e-074,
            7.55581902e-086,
            1.92194773e-098,
            6.61626106e-112,
            3.08244070e-126,
            1.94351485e-141,
            1.65841048e-157,
            1.91516960e-174,
        ]
    )
    assert np.allclose(p_fa, p_fa_truth)


def test_real_square_law():
    threshold = np.arange(0, 21)
    sigma2 = 1
    p_fa = sdr.p_fa(threshold, sigma2, detector="square-law", complex=False)
    p_fa_truth = np.array(
        [
            [
                1.00000000e00,
                3.17310508e-01,
                1.57299207e-01,
                8.32645167e-02,
                4.55002639e-02,
                2.53473187e-02,
                1.43058784e-02,
                8.15097159e-03,
                4.67773498e-03,
                2.69979606e-03,
                1.56540226e-03,
                9.11118877e-04,
                5.32005505e-04,
                3.11490977e-04,
                1.82810633e-04,
                1.07511177e-04,
                6.33424837e-05,
                3.73798184e-05,
                2.20904970e-05,
                1.30718454e-05,
                7.74421643e-06,
            ]
        ]
    )
    assert np.allclose(p_fa, p_fa_truth)


def test_complex_square_law():
    threshold = np.arange(0, 21)
    sigma2 = 1
    p_fa = sdr.p_fa(threshold, sigma2, detector="square-law")
    p_fa_truth = np.array(
        [
            1.00000000e00,
            3.67879441e-01,
            1.35335283e-01,
            4.97870684e-02,
            1.83156389e-02,
            6.73794700e-03,
            2.47875218e-03,
            9.11881966e-04,
            3.35462628e-04,
            1.23409804e-04,
            4.53999298e-05,
            1.67017008e-05,
            6.14421235e-06,
            2.26032941e-06,
            8.31528719e-07,
            3.05902321e-07,
            1.12535175e-07,
            4.13993772e-08,
            1.52299797e-08,
            5.60279644e-09,
            2.06115362e-09,
        ]
    )
    assert np.allclose(p_fa, p_fa_truth)
