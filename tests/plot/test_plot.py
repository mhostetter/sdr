import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.testing.compare
import numpy as np
import scipy.signal

import sdr

sdr.plot.use_style()

WRITE_TRUTH = False
TEST_FOLDER = Path(tempfile.gettempdir())
TRUTH_FOLDER = Path(__file__).parent


def test_time_domain_y_only(request):
    test_filename = (TEST_FOLDER / request.node.name).with_suffix(".png")
    truth_filename = (TRUTH_FOLDER / request.node.name).with_suffix(".png")

    N = 1_000  # Number of samples
    T = 0.01  # Sample time
    t = np.arange(N) * T  # Time vector
    x_lin = scipy.signal.chirp(t, f0=6, f1=1, t1=10, method="linear")

    plt.figure()
    sdr.plot.time_domain(x_lin)
    plt.title("Linear chirp signal")
    if WRITE_TRUTH:
        plt.savefig(truth_filename)
    plt.savefig(test_filename)
    plt.close()

    diff = matplotlib.testing.compare.compare_images(truth_filename, test_filename, tol=0)
    assert diff is None


def test_time_domain_x_and_y(request):
    test_filename = (TEST_FOLDER / request.node.name).with_suffix(".png")
    truth_filename = (TRUTH_FOLDER / request.node.name).with_suffix(".png")

    N = 1_000  # Number of samples
    T = 0.01  # Sample time
    t = np.arange(N) * T  # Time vector
    x_lin = scipy.signal.chirp(t, f0=6, f1=1, t1=10, method="linear")

    plt.figure()
    sdr.plot.time_domain(t, x_lin, sample_rate=1 / T)
    plt.title("Linear chirp signal")
    if WRITE_TRUTH:
        plt.savefig(truth_filename)
    plt.savefig(test_filename)
    plt.close()

    diff = matplotlib.testing.compare.compare_images(truth_filename, test_filename, tol=0)
    assert diff is None
