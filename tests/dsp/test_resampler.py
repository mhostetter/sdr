import numpy as np

import sdr


def debug_plot(y, y_truth):
    import matplotlib.pyplot as plt

    plt.figure()
    sdr.plot.time_domain(y, label="Test")
    sdr.plot.time_domain(y_truth, label="Truth")
    plt.legend()
    plt.show()


# def test_non_streaming_rate():
#     mode = "rate"
#     N = 50
#     x = np.random.randn(N) + 1j * np.random.randn(N)  # Input signal
#     r = np.random.randint(3, 7)  # Interpolation rate

#     fir = sdr.Interpolator(r)
#     y = fir(x, mode)

#     # The output should align with the input. Every r-th sample should match.
#     np.testing.assert_array_almost_equal(y[::r], x)


# def test_non_streaming_full():
#     mode = "full"
#     N = 50
#     x = np.random.randn(N) + 1j * np.random.randn(N)  # Input signal
#     r = np.random.randint(3, 7)  # Interpolation rate

#     fir = sdr.Interpolator(r)
#     y = fir(x, mode)

#     xr = np.zeros(N * r, dtype=complex)
#     xr[::r] = x[:]
#     y_truth = scipy.signal.convolve(xr, fir.taps, mode=mode)

#     # Given the polyphase decomposition, the polyphase output is slightly shorter
#     np.testing.assert_array_almost_equal(y, y_truth[: y.size])


# def test_streaming():
#     N = 50
#     x = np.random.randn(N) + 1j * np.random.randn(N)  # Input signal
#     r = np.random.randint(3, 7)  # Interpolation rate

#     fir = sdr.Interpolator(r, streaming=True)

#     d = 10  # Stride
#     y = np.zeros(N * r, dtype=complex)
#     for i in range(0, N, d):
#         y[i * r : (i + d) * r] = fir(x[i : i + d])

#     xr = np.zeros(N * r, dtype=complex)
#     xr[::r] = x[:]
#     y_truth = scipy.signal.convolve(xr, fir.taps, mode="full")[0 : N * r]

#     np.testing.assert_array_almost_equal(y, y_truth)


# def test_streaming_match_full():
#     N = 50
#     x = np.random.randn(N) + 1j * np.random.randn(N)  # Input signal
#     r = np.random.randint(3, 7)  # Interpolation rate

#     fir1 = sdr.Interpolator(r)
#     y_full = fir1(x, mode="full")

#     fir2 = sdr.Interpolator(r, streaming=True)
#     d = 10  # Stride
#     y_stream = np.zeros_like(y_full)
#     for i in range(0, N, d):
#         y_stream[i * r : (i + d) * r] = fir2(x[i : i + d])
#     y_stream[(i + d) * r :] = fir2.flush()

#     np.testing.assert_array_almost_equal(y_full, y_stream)


def test_7_5_kaiser():
    """
    MATLAB:
        >> x = cos(pi/4*(0:19)');
        >> fir = dsp.FIRRateConverter(7, 5);
        >> y = fir(x);
    """
    x = np.cos(np.pi / 4 * np.arange(20))
    y_truth = np.array(
        [
            0,
            -0.000221371923965,
            0.000574219878925,
            -0.000488015063336,
            -0.000845633199504,
            0.003230097703218,
            -0.004016421787642,
            0,
            0.008762301554564,
            -0.015547832246794,
            0.009668969889023,
            0.013408679740684,
            -0.042210721138260,
            0.048500035713907,
            0,
            -0.129289109639484,
            0.426798688747611,
            1.065732496320718,
            0.822489045403122,
            0.270415487739051,
            -0.189056975480333,
            -0.707106781186547,
            -0.992176816644380,
            -0.928552056386907,
            -0.628285559938961,
            -0.115259551360420,
            0.438835907100540,
            0.844204496412674,
        ]
    )

    fir = sdr.Resampler(7, 5, streaming=True)
    y = fir(x)

    # debug_plot(y, y_truth)
    np.testing.assert_almost_equal(y, y_truth)


def test_7_5_linear():
    """
    MATLAB:
        >> x = cos(pi/4*(0:19)');
        >> fir = dsp.FIRRateConverter(7, 5, 'linear');
        >> y = fir(x);
    """
    x = np.cos(np.pi / 4 * np.arange(20))
    y_truth = np.array(
        [
            0.142857142857143,
            0.857142857142857,
            0.832632446392313,
            0.505076272276105,
            0.000000000000000,
            -0.505076272276105,
            -0.832632446392313,
            -0.958158111598078,
            -0.748948669588469,
            -0.303045763365663,
            0.202030508910442,
            0.707106781186547,
            0.916316223196156,
            0.874474334794235,
            0.606091526731327,
            0.101015254455221,
            -0.404061017820884,
            -0.790790557990390,
            -1.000000000000000,
            -0.790790557990391,
            -0.404061017820884,
            0.101015254455221,
            0.606091526731325,
            0.874474334794234,
            0.916316223196156,
            0.707106781186547,
            0.202030508910442,
            -0.303045763365662,
        ]
    )

    fir = sdr.Resampler(7, 5, "linear-matlab", streaming=True)
    y = fir(x)

    # debug_plot(y, y_truth)
    np.testing.assert_almost_equal(y, y_truth)


def test_7_5_zoh():
    """
    MATLAB:
        >> x = cos(pi/4*(0:19)');
        >> fir = dsp.FIRRateConverter(7, 5, 'zoh');
        >> y = fir(x);
    """
    x = np.cos(np.pi / 4 * np.arange(20))
    y_truth = np.array(
        [
            1.000000000000000,
            1.000000000000000,
            0.707106781186548,
            0.000000000000000,
            0.000000000000000,
            -0.707106781186547,
            -1.000000000000000,
            -0.707106781186548,
            -0.707106781186548,
            -0.000000000000000,
            0.707106781186547,
            0.707106781186547,
            1.000000000000000,
            0.707106781186548,
            0.000000000000000,
            0.000000000000000,
            -0.707106781186547,
            -1.000000000000000,
            -1.000000000000000,
            -0.707106781186547,
            -0.000000000000000,
            0.707106781186547,
            0.707106781186547,
            1.000000000000000,
            0.707106781186547,
            0.707106781186547,
            0.000000000000001,
            -0.707106781186546,
        ]
    )

    fir = sdr.Resampler(7, 5, "zoh", streaming=True)
    y = fir(x)

    # debug_plot(y, y_truth)
    np.testing.assert_almost_equal(y, y_truth)


def test_5_7_kaiser():
    """
    MATLAB:
        >> x = cos(pi/4*(0:20)');
        >> fir = dsp.FIRRateConverter(5, 7);
        >> y = fir(x);
    """
    x = np.cos(np.pi / 4 * np.arange(20))
    y_truth = np.array(
        [
            0.000069187197637,
            -0.000581099350198,
            0.001968738750881,
            -0.004794480014143,
            0.010241960638799,
            -0.019969377248886,
            0.037900876807341,
            -0.079345721289059,
            0.286007581662442,
            0.964071849392977,
            -0.034189330257715,
            -0.873227486664808,
            -0.818051016895984,
            0.160578164604500,
            0.949478110003271,
        ]
    )

    fir = sdr.Resampler(5, 7, streaming=True)
    y = fir(x)

    # debug_plot(y, y_truth)
    np.testing.assert_almost_equal(y, y_truth)


def test_5_7_linear():
    """
    MATLAB:
        >> x = cos(pi/4*(0:20)');
        >> fir = dsp.FIRRateConverter(5, 7, 'linear');
        >> y = fir(x);
    """
    x = np.cos(np.pi / 4 * np.arange(20))
    y_truth = np.array(
        [
            0.200000000000000,
            0.824264068711929,
            0.000000000000000,
            -0.824264068711928,
            -0.765685424949238,
            0.141421356237309,
            0.882842712474619,
            0.707106781186548,
            -0.282842712474619,
            -0.941421356237309,
            -0.565685424949238,
            0.424264068711928,
            1.000000000000000,
            0.424264068711929,
            -0.565685424949237,
        ]
    )

    fir = sdr.Resampler(5, 7, "linear-matlab", streaming=True)
    y = fir(x)

    # debug_plot(y, y_truth)
    np.testing.assert_almost_equal(y, y_truth)


def test_5_7_zoh():
    """
    MATLAB:
        >> x = cos(pi/4*(0:20)');
        >> fir = dsp.FIRRateConverter(5, 7, 'zoh');
        >> y = fir(x);
    """
    x = np.cos(np.pi / 4 * np.arange(20))
    y_truth = np.array(
        [
            1.000000000000000,
            0.707106781186548,
            0.000000000000000,
            -1.000000000000000,
            -0.707106781186548,
            0.707106781186547,
            1.000000000000000,
            0.707106781186548,
            -0.707106781186547,
            -1.000000000000000,
            -0.000000000000000,
            0.707106781186547,
            1.000000000000000,
            0.000000000000001,
            -0.707106781186546,
        ]
    )

    fir = sdr.Resampler(5, 7, "zoh", streaming=True)
    y = fir(x)

    # debug_plot(y, y_truth)
    np.testing.assert_almost_equal(y, y_truth)
