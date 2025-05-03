import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import Literal

import sdr


def test_linear_taps():
    """
    MATLAB:
        >> frc = dsp.FarrowRateConverter(PolynomialOrder=1)
        >> getPolynomialCoefficients(frc)
    """
    farrow = sdr.FarrowFractionalDelay(1)
    taps = farrow.taps
    taps_truth = np.array(
        [
            [-1, 1],
            [1, 0],
        ]
    ).T
    assert taps.shape == taps_truth.shape
    assert np.allclose(taps, taps_truth)


def test_quadratic_taps():
    """
    MATLAB:
        >> frc = dsp.FarrowRateConverter(PolynomialOrder=2)
        >> getPolynomialCoefficients(frc)
    """
    farrow = sdr.FarrowFractionalDelay(2)
    taps = farrow.taps
    taps_truth = np.array(
        [
            [1 / 2, -1 / 2, 0],
            [-1, 0, 1],
            [1 / 2, 1 / 2, 0],
        ]
    ).T
    assert taps.shape == taps_truth.shape
    assert np.allclose(taps, taps_truth)


def test_cubic_taps():
    """
    MATLAB:
        >> frc = dsp.FarrowRateConverter(PolynomialOrder=3)
        >> getPolynomialCoefficients(frc)
    """
    farrow = sdr.FarrowFractionalDelay(3)
    taps = farrow.taps
    taps_truth = np.array(
        [
            [-1 / 6, 1 / 2, -1 / 3, 0],
            [1 / 2, -1, -1 / 2, 1],
            [-1 / 2, 1 / 2, 1, 0],
            [1 / 6, 0, -1 / 6, 0],
        ]
    ).T
    assert taps.shape == taps_truth.shape
    assert np.allclose(taps, taps_truth)


def test_quartic_taps():
    """
    MATLAB:
        >> frc = dsp.FarrowRateConverter(PolynomialOrder=4)
        >> getPolynomialCoefficients(frc)
    """
    farrow = sdr.FarrowFractionalDelay(4)
    taps = farrow.taps
    taps_truth = np.array(
        [
            [1 / 24, -1 / 12, -1 / 24, 1 / 12, 0],
            [-1 / 6, 1 / 6, 2 / 3, -2 / 3, 0],
            [1 / 4, 0, -5 / 4, 0, 1],
            [-1 / 6, -1 / 6, 2 / 3, 2 / 3, 0],
            [1 / 24, 1 / 12, -1 / 24, -1 / 12, 0],
        ]
    ).T
    assert taps.shape == taps_truth.shape
    assert np.allclose(taps, taps_truth)


# def test_cubic_delay_0p25():
#     """
#     MATLAB:
#         >> x = transpose(cos(2 * pi / 10 .* (0:40)));
#         >> vfd = dsp.VariableFractionalDelay(InterpolationMethod='Farrow', FilterLength=3+1);
#         >> y = vfd(x, 0.25); y
#     """
#     mu = 0.25
#     x = np.cos(2 * np.pi / 10 * np.arange(41))
#     y_truth = np.array(
#         [
#             0.601562500000000,
#             1.088236785678679,
#             0.414754821357359,
#             -0.153887193862291,
#             -0.707992648339770,
#             -0.991668974936508,
#             -0.896561458696268,
#             -0.458997938237239,
#             0.153887193862291,
#             0.707992648339770,
#             0.991668974936508,
#             0.896561458696268,
#             0.458997938237239,
#             -0.153887193862291,
#             -0.707992648339769,
#             -0.991668974936508,
#             -0.896561458696268,
#             -0.458997938237239,
#             0.153887193862291,
#             0.707992648339769,
#             0.991668974936508,
#             0.896561458696268,
#             0.458997938237239,
#             -0.153887193862290,
#             -0.707992648339769,
#             -0.991668974936508,
#             -0.896561458696268,
#             -0.458997938237239,
#             0.153887193862290,
#             0.707992648339769,
#             0.991668974936508,
#             0.896561458696268,
#             0.458997938237239,
#             -0.153887193862290,
#             -0.707992648339769,
#             -0.991668974936508,
#             -0.896561458696268,
#             -0.458997938237239,
#             0.153887193862290,
#             0.707992648339769,
#             0.991668974936508,
#         ]
#     )
#     verify_output_single(1, mu, x, y_truth)
#     verify_output_multiple(1, mu, x, y_truth)


# def test_cubic_delay_0p5():
#     """
#     MATLAB:
#         >> x = transpose(cos(2 * pi / 10 .* (0:40)));
#         >> vfd = dsp.VariableFractionalDelay(InterpolationMethod='Farrow', FilterLength=3+1);
#         >> y = vfd(x, 0.5); y
#     """
#     mu = 0.5
#     x = np.cos(2 * np.pi / 10 * np.arange(41))
#     y_truth = np.array(
#         [
#             0,
#             1.000000000000000,
#             0.809016994374947,
#             0.309016994374947,
#             -0.309016994374947,
#             -0.809016994374947,
#             -1.000000000000000,
#             -0.809016994374947,
#             -0.309016994374948,
#             0.309016994374947,
#             0.809016994374947,
#             1.000000000000000,
#             0.809016994374948,
#             0.309016994374948,
#             -0.309016994374947,
#             -0.809016994374947,
#             -1.000000000000000,
#             -0.809016994374948,
#             -0.309016994374948,
#             0.309016994374947,
#             0.809016994374947,
#             1.000000000000000,
#             0.809016994374948,
#             0.309016994374948,
#             -0.309016994374947,
#             -0.809016994374947,
#             -1.000000000000000,
#             -0.809016994374948,
#             -0.309016994374948,
#             0.309016994374947,
#             0.809016994374947,
#             1.000000000000000,
#             0.809016994374948,
#             0.309016994374948,
#             -0.309016994374947,
#             -0.809016994374947,
#             -1.000000000000000,
#             -0.809016994374948,
#             -0.309016994374948,
#             0.309016994374947,
#             0.809016994374947,
#         ]
#     )
#     verify_output_single(1, mu, x, y_truth)
#     verify_output_multiple(1, mu, x, y_truth)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("mu", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("mode", ["rate", "full"])
def test_modes_constant_mu(order, mu, mode):
    sps = 10
    span = 4
    x = sdr.root_raised_cosine(0.5, span, sps, norm="power")

    compare_modes(order, x, mu, mode)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("mode", ["rate", "full"])
def test_modes_linear_ramp_mu(order, mode):
    sps = 10
    span = 4
    x = sdr.root_raised_cosine(0.5, span, sps, norm="power")
    mu = np.linspace(0, 1, x.size)

    compare_modes(order, x, mu, mode)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("mode", ["rate", "full"])
def test_modes_random_mu(order, mode):
    sps = 10
    span = 4
    x = sdr.root_raised_cosine(0.5, span, sps, norm="power")
    mu = np.random.default_rng().uniform(0, 1, x.size)

    compare_modes(order, x, mu, mode, stride=16)


def compare_modes(order: int, x: npt.NDArray, mu: npt.ArrayLike, mode: Literal["rate", "full"], stride: int = 10):
    x, mu = np.broadcast_arrays(x, mu)

    # Non-streaming
    farrow = sdr.FarrowFractionalDelay(order, streaming=False)
    y_ns = farrow(x, mu=mu, mode=mode)

    # Streaming
    farrow = sdr.FarrowFractionalDelay(order, streaming=True)
    y = []
    for i in range(0, x.size, stride):
        yi = farrow(x[i : i + stride], mu=mu[i : i + stride], mode=mode)
        y.append(yi)
    # y.append(farrow.flush())  # Need to flush the filter state
    y_s = np.concatenate(y)

    if False:
        plt.figure()
        sdr.plot.time_domain(x, label="Input")
        sdr.plot.time_domain(y_ns, marker="o", fillstyle="none", label="Non-streaming")
        sdr.plot.time_domain(y_s, marker=".", label="Streaming")
        plt.title(f"Farrow Fractional Delay (order={order}, mu={mu[0]})")
        plt.show()

    assert np.allclose(y_ns, y_s)


def debug_plot(x: np.ndarray, y: np.ndarray, y_truth: np.ndarray, offset: float):
    plt.figure()
    sdr.plot.time_domain(x, marker=".", label="x")
    sdr.plot.time_domain(y_truth, offset=-offset, marker="o", label="y_truth")
    sdr.plot.time_domain(y, offset=-offset, marker="x", label="y")
    plt.legend()
    plt.show()


def verify_output_single(order: int, mu: float, x: np.ndarray, y_truth: np.ndarray):
    farrow = sdr.FarrowFractionalDelay(order, streaming=True)
    y = farrow(x, mu=mu)
    # debug_plot(x, y, y_truth, farrow._delay)
    assert np.allclose(y, y_truth)


def verify_output_multiple(order: int, mu: float, x: np.ndarray, y_truth: np.ndarray):
    farrow = sdr.FarrowFractionalDelay(order, streaming=True)
    ys = []
    for i in range(0, x.size, 10):
        yi = farrow(x[i : i + 10], mu=mu)
        ys.append(yi)
    y = np.concatenate(ys)
    # debug_plot(x, y, y_truth, farrow._delay)
    assert np.allclose(y, y_truth)
