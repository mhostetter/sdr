import numpy as np

import sdr


def test_vector():
    """
    These numbers were verified visually from published papers.
    """
    snr = np.linspace(-30, 20, 11)
    C = sdr.biawgn_capacity(snr)
    C_truth = np.array(
        [
            7.20987087e-04,
            2.27750199e-03,
            7.17764533e-03,
            2.24576590e-02,
            6.87433134e-02,
            1.97731546e-01,
            4.85944154e-01,
            8.59194084e-01,
            9.96756328e-01,
            9.99999959e-01,
            1.00000000e00,
        ]
    )
    assert np.allclose(C, C_truth)
