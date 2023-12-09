import numpy as np

import sdr


def test_da_error():
    A_rx, A_ref = 5, 3
    ped = sdr.MLPED(A_rx, A_ref)
    modem = sdr.PSK(4)
    error, da_error = ped.data_aided_error(modem)
    da_error_truth = A_rx * A_ref * np.sin(error)
    assert np.allclose(da_error, da_error_truth)
