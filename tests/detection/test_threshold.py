"""
These test vectors were manually verified through simulation.
"""
import numpy as np

import sdr


def test_real_coherent():
    p_fa = np.logspace(-12, -1, 12)
    sigma2 = 1
    threshold = sdr.threshold(p_fa, sigma2, detector="coherent", complex=False)
    threshold_truth = np.array(
        [
            [
                7.03448383,
                6.70602316,
                6.3613409,
                5.99780702,
                5.61200124,
                5.19933758,
                4.75342431,
                4.26489079,
                3.71901649,
                3.09023231,
                2.32634787,
                1.28155157,
            ]
        ]
    )
    assert np.allclose(threshold, threshold_truth)


def test_complex_coherent():
    p_fa = np.logspace(-12, -1, 12)
    sigma2 = 1
    threshold = sdr.threshold(p_fa, sigma2, detector="coherent")
    threshold_truth = np.array(
        [
            4.97413122,
            4.74187445,
            4.49814729,
            4.24109001,
            3.96828414,
            3.67648686,
            3.36117856,
            3.0157332,
            2.62974178,
            2.18512422,
            1.64497636,
            0.9061938,
        ]
    )
    assert np.allclose(threshold, threshold_truth)


def test_real_linear():
    p_fa = np.logspace(-12, -1, 12)
    sigma2 = 1
    threshold = sdr.threshold(p_fa, sigma2, detector="linear", complex=False)
    threshold_truth = np.array(
        [
            7.13050685,
            6.80650249,
            6.46695109,
            6.1094102,
            5.73072887,
            5.32672389,
            4.89163848,
            4.41717341,
            3.89059189,
            3.29052673,
            2.5758293,
            1.64485363,
        ]
    )
    assert np.allclose(threshold, threshold_truth)


def test_complex_linear():
    p_fa = np.logspace(-12, -1, 12)
    sigma2 = 1
    threshold = sdr.threshold(p_fa, sigma2, detector="linear")
    threshold_truth = np.array(
        [
            5.25652177,
            5.03273643,
            4.79852591,
            4.55228139,
            4.29193205,
            4.01473482,
            3.71692219,
            3.39307021,
            3.03485426,
            2.62826088,
            2.14596603,
            1.51742713,
        ]
    )
    assert np.allclose(threshold, threshold_truth)


def test_real_square_law():
    p_fa = np.logspace(-12, -1, 12)
    sigma2 = 1
    threshold = sdr.threshold(p_fa, sigma2, detector="square-law", complex=False)
    threshold_truth = np.array(
        [
            [
                50.84412791,
                46.32847616,
                41.82145636,
                37.32489305,
                32.84125336,
                28.37398736,
                23.92812698,
                19.51142096,
                15.13670523,
                10.82756617,
                6.6348966,
                2.70554345,
            ]
        ]
    )
    assert np.allclose(threshold, threshold_truth)


def test_complex_square_law():
    p_fa = np.logspace(-12, -1, 12)
    sigma2 = 1
    threshold = sdr.threshold(p_fa, sigma2, detector="square-law")
    threshold_truth = np.array(
        [
            27.63102112,
            25.32843602,
            23.02585093,
            20.72326584,
            18.42068074,
            16.11809565,
            13.81551056,
            11.51292546,
            9.21034037,
            6.90775528,
            4.60517019,
            2.30258509,
        ]
    )
    assert np.allclose(threshold, threshold_truth)
