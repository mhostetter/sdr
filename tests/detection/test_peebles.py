"""
The equation was compared against plots from a technical paper. These values were then calculated once using the
"verified" code.
"""

import sdr


def test_0p5():
    p_d = 0.5
    p_fa = 1e-2
    assert sdr.peebles(p_d, p_fa, 1) == 0.0
    assert sdr.peebles(p_d, p_fa, 2) == 2.3041591328608138
    assert sdr.peebles(p_d, p_fa, 3) == 3.5677189691672773


def test_0p9():
    p_d = 0.9
    p_fa = 1e-6
    assert sdr.peebles(p_d, p_fa, 1) == 0.0
    assert sdr.peebles(p_d, p_fa, 2) == 2.717834481430985
    assert sdr.peebles(p_d, p_fa, 3) == 4.208246512218635


def test_0p999():
    p_d = 0.999
    p_fa = 1e-10
    assert sdr.peebles(p_d, p_fa, 1) == 0.0
    assert sdr.peebles(p_d, p_fa, 2) == 2.984178567081324
    assert sdr.peebles(p_d, p_fa, 3) == 4.620648951420142
