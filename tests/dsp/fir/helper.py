import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import sdr


def debug_plot(h: npt.NDArray, h_truth: npt.NDArray):
    plt.figure()
    sdr.plot.time_domain(h_truth, marker=".", label="h_truth")
    sdr.plot.time_domain(h, marker="x", label="h")
    plt.legend()

    plt.figure()
    sdr.plot.magnitude_response(h_truth, label="h_truth")
    sdr.plot.magnitude_response(h, label="h")
    plt.legend()

    plt.show()


def verify_impulse_response(h: npt.NDArray, h_truth: npt.NDArray, atol: float = 1e-6):
    # debug_plot(h, h_truth)

    # MATLAB sets the center of the passband to 0 dB gain. We set the average passband gain to 0 dB.
    # Normalize the impulse response to have the same RMS voltage as MATLAB for comparison.
    h *= sdr.rms_voltage(h_truth) / sdr.rms_voltage(h)

    assert h.shape == h_truth.shape
    assert np.allclose(h, h_truth, atol=atol)
