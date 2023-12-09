import numpy as np

import sdr


def test_da_error():
    ped = sdr.PED()
    modem = sdr.PSK(4)
    error, da_error = ped.data_aided_error(modem)
    da_error_truth = error
    assert np.allclose(da_error, da_error_truth)
