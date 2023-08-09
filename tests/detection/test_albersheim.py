"""
disp('Vary pd')
for pd = 0.05:0.05:0.95
    snr = albersheim(pd, 1e-6, 1);
    disp(snr)
end
disp('Vary pfa')
for power = -15:1:-1
    snr = albersheim(0.5, 10^power, 1);
    disp(snr)
end
disp('Vary Nnc')
for Nnc = [1 5 10 15 20 25 30 40 50 60 70 80 90 100 200 500 1000]
    snr = albersheim(0.5, 10^-6, Nnc);
    disp(snr)
end
"""
import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # p_d must be in (0, 1)
        sdr.albersheim(0, 1e-6, 1)
    with pytest.raises(ValueError):
        # p_d must be in (0, 1)
        sdr.albersheim(1, 1e-6, 1)
    with pytest.raises(ValueError):
        # p_fa must be in (0, 1)
        sdr.albersheim(0.5, 0, 1)
    with pytest.raises(ValueError):
        # p_fa must be in (0, 1)
        sdr.albersheim(0.5, 1, 1)
    with pytest.raises(ValueError):
        # N_nc must be at least 1
        sdr.albersheim(0.5, 1e-6, 0)


def test_across_p_d():
    p_d = np.arange(0.05, 1, 0.05)
    snr = sdr.albersheim(p_d, 1e-6, 1)
    snr_truth = np.array(
        [
            5.577010678961301,
            7.829920769472949,
            8.800540626254417,
            9.410329568464647,
            9.856412045680429,
            10.211758066019740,
            10.511060019786157,
            10.773550923327434,
            11.011152537700355,
            11.231984877880109,
            11.442112499093184,
            11.646548224380126,
            11.849953212372384,
            12.057285789014941,
            12.274647700594016,
            12.510781410840821,
            12.780466849937795,
            13.114544494322786,
            13.605048844272062,
        ]
    )
    np.testing.assert_allclose(snr, snr_truth)


def test_across_p_fa():
    p_fa = 10.0 ** np.arange(-15, 0)
    snr = sdr.albersheim(0.5, p_fa, 1)
    snr_truth = np.array(
        [
            15.297003299268226,
            14.993521787461759,
            14.667188248832815,
            14.314278887624823,
            13.930078319024233,
            13.508492912451231,
            13.041455112681206,
            12.517966524335804,
            11.922480132961311,
            11.231984877880109,
            10.410300484247330,
            9.395603267835611,
            8.068450388126450,
            6.146225293027499,
            2.607203414307889,
        ]
    )
    np.testing.assert_allclose(snr, snr_truth)


def test_across_N_nc():
    N_nc = np.array([1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000])
    snr = sdr.albersheim(0.5, 1e-6, N_nc)
    snr_truth = np.array(
        [
            11.231984877880109,
            5.670572205727751,
            3.556291380845494,
            2.394909665228113,
            1.600092813526487,
            0.998450342864260,
            0.515644335269712,
            -0.231629668145274,
            -0.800194343455530,
            -1.258285958879696,
            -1.641440584682812,
            -1.970484329329963,
            -2.258654478208143,
            -2.514880224697833,
            -4.168911815913020,
            -6.291064789951179,
            -7.863055239050709,
        ]
    )
    np.testing.assert_allclose(snr, snr_truth)
