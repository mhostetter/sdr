"""
disp('Vary pd')
for pd = 0.1:0.05:0.95
    snr = shnidman(pd, 1e-6, 1);
    disp(snr)
end
disp('Vary pfa')
for power = -15:1:-1
    snr = shnidman(0.5, 10^power, 1);
    disp(snr)
end
disp('Vary Nnc')
for Nnc = [1 5 10 15 20 25 30 40 50 60 70 80 90 100 200 500 1000]
    snr = shnidman(0.5, 10^-6, Nnc);
    disp(snr)
end
"""

import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # p_d must be in (0, 1)
        sdr.shnidman(0, 1e-6, 1)
    with pytest.raises(ValueError):
        # p_d must be in (0, 1)
        sdr.shnidman(1, 1e-6, 1)
    with pytest.raises(ValueError):
        # p_fa must be in (0, 1)
        sdr.shnidman(0.5, 0, 1)
    with pytest.raises(ValueError):
        # p_fa must be in (0, 1)
        sdr.shnidman(0.5, 1, 1)
    with pytest.raises(ValueError):
        # n_nc must be at least 1
        sdr.shnidman(0.5, 1e-6, 0)


def test_across_p_d():
    p_d = np.arange(0.1, 1, 0.05)
    snr = sdr.shnidman(p_d, 1e-6, 1)
    snr_truth = np.array(
        [
            8.638219944424179,
            9.176472527195301,
            9.584618391346567,
            9.921817222132290,
            10.214925964858082,
            10.478648752587219,
            10.722106546144460,
            10.951539545173148,
            11.171613923710709,
            11.386150421902641,
            11.598604211020529,
            11.812462984419767,
            12.031689396435866,
            12.261376243923564,
            12.508971120024171,
            12.787083940503294,
            13.121692696758089,
            13.588328948705623,
        ]
    )
    np.testing.assert_allclose(snr, snr_truth)


def test_across_p_fa():
    p_fa = 10.0 ** np.arange(-15, 0)
    snr = sdr.shnidman(0.5, p_fa, 1)
    snr_truth = np.array(
        [
            15.006746068762709,
            14.719925630256057,
            14.411725015222657,
            14.078648305868455,
            13.716257825222556,
            13.318800181756899,
            12.878625161335435,
            12.385241849116994,
            11.823699690860849,
            11.171613923710709,
            10.393186598006821,
            9.425625773718195,
            8.142386493822812,
            6.222509416067624,
            2.358765182107788,
        ]
    )
    np.testing.assert_allclose(snr, snr_truth)


def test_across_n_nc():
    n_nc = np.array([1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000])
    snr = sdr.shnidman(0.5, 1e-6, n_nc)
    snr_truth = np.array(
        [
            11.171613923710709,
            5.889057826046667,
            3.745346308014405,
            2.540560034820032,
            1.707866317562093,
            1.074345861541872,
            0.564527897904441,
            -0.226123038305275,
            -0.811680151206701,
            -1.299262221370157,
            -1.706607568412664,
            -2.055994441685868,
            -2.361604728972054,
            -2.633013225970103,
            -4.376499978073298,
            -6.589223941137504,
            -8.211273144667473,
        ]
    )
    np.testing.assert_allclose(snr, snr_truth)
