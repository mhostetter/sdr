import pytest

import sdr


def test_ppi_gains():
    """
    Reference:
        - M. Rice, Digital Communications: A Discrete-Time Approach, Appendix C.2.1.
    """
    lf = sdr.LoopFilter(0.05, 1)
    assert lf.K1 == pytest.approx(0.1479, rel=3)
    assert lf.K2 == pytest.approx(0.0059, rel=3)
