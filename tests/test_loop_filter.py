import numpy as np
import pytest

import sdr


def test_ppi_gains_1():
    """
    Reference:
        - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Section 7.2.3 Example #1
    """
    BnT = 0.02  # Normalized noise bandwidth
    zeta = 1 / np.sqrt(2)  # Damping factor
    K0 = 1  # NCO gain
    Kp = 2  # PED gain
    lf = sdr.LoopFilter(BnT, zeta, K0=K0, Kp=Kp)
    assert lf.K1 == pytest.approx(2.6e-2, rel=3)
    assert lf.K2 == pytest.approx(6.9e-4, rel=3)


def test_ppi_gains_2():
    """
    Reference:
        - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Section 7.2.3 Example #2
    """
    BnT = 0.02 / 16  # Normalized noise bandwidth
    zeta = 1 / np.sqrt(2)  # Damping factor
    K0 = 16  # NCO gain
    Kp = 2  # PED gain
    lf = sdr.LoopFilter(BnT, zeta, K0=K0, Kp=Kp)
    assert lf.K1 == pytest.approx(1.7e-3, rel=3)
    assert lf.K2 == pytest.approx(2.8e-6, rel=3)


def test_ppi_gains_3():
    """
    Reference:
        - Michael Rice, *Digital Communications: A Discrete-Time Approach*, Appendix C.2.1.
    """
    BnT = 0.05  # Normalized noise bandwidth
    zeta = 1  # Damping factor
    K0 = 1  # NCO gain
    Kp = 1  # PED gain
    lf = sdr.LoopFilter(BnT, zeta, K0=K0, Kp=Kp)
    assert lf.K1 == pytest.approx(0.1479, rel=3)
    assert lf.K2 == pytest.approx(0.0059, rel=3)
