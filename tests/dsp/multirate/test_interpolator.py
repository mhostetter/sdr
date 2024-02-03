import numpy as np
import scipy.signal

import sdr


def debug_plot(y, y_truth):
    import matplotlib.pyplot as plt

    plt.figure()
    sdr.plot.time_domain(y, label="Test")
    sdr.plot.time_domain(y_truth, label="Truth")
    plt.legend()
    plt.show()


def test_non_streaming_rate():
    mode = "rate"
    N = 50
    rng = np.random.default_rng()
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)  # Input signal
    r = rng.integers(3, 7)  # Interpolation rate

    fir = sdr.Interpolator(r)
    y = fir(x, mode)

    # The output should align with the input. Every r-th sample should match.
    y = y[::r]
    y_truth = x

    # debug_plot(y, y_truth)
    np.testing.assert_array_almost_equal(y, y_truth)


def test_non_streaming_full():
    mode = "full"
    N = 50
    rng = np.random.default_rng()
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)  # Input signal
    r = rng.integers(3, 7)  # Interpolation rate

    fir = sdr.Interpolator(r)
    y = fir(x, mode)

    xr = np.zeros(N * r, dtype=complex)
    xr[::r] = x[:]
    # Given the polyphase decomposition, the polyphase output is slightly shorter
    y_truth = scipy.signal.convolve(xr, fir.taps, mode=mode)[: y.size]

    # debug_plot(y, y_truth)
    np.testing.assert_array_almost_equal(y, y_truth)


def test_streaming():
    N = 50
    rng = np.random.default_rng()
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)  # Input signal
    r = rng.integers(3, 7)  # Interpolation rate

    fir = sdr.Interpolator(r, streaming=True)

    d = 10  # Stride
    y = np.zeros(N * r, dtype=complex)
    for i in range(0, N, d):
        y[i * r : (i + d) * r] = fir(x[i : i + d])

    xr = np.zeros(N * r, dtype=complex)
    xr[::r] = x[:]
    y_truth = scipy.signal.convolve(xr, fir.taps, mode="full")[0 : N * r]

    # debug_plot(y, y_truth)
    np.testing.assert_array_almost_equal(y, y_truth)


def test_streaming_match_full():
    N = 50
    rng = np.random.default_rng()
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)  # Input signal
    r = rng.integers(3, 7)  # Interpolation rate

    fir1 = sdr.Interpolator(r)
    y_full = fir1(x, mode="full")

    fir2 = sdr.Interpolator(r, streaming=True)
    d = 10  # Stride
    y_stream = np.zeros_like(y_full)
    for i in range(0, N, d):
        y_stream[i * r : (i + d) * r] = fir2(x[i : i + d])
    y_stream[(i + d) * r :] = fir2.flush()

    np.testing.assert_array_almost_equal(y_full, y_stream)


def test_3_kaiser():
    """
    MATLAB:
        >> x = cos(pi/4*(0:19)');
        >> fir = dsp.Interpolator(3);
        >> y = fir(x);
    """
    x = np.cos(np.pi / 4 * np.arange(20))
    y_truth = np.array(
        [
            0,
            -0.000129061486200,
            -0.000228040316281,
            0,
            0.000463353486490,
            0.000641358576241,
            0,
            -0.001136952367305,
            -0.001468858650183,
            0,
            0.002432313713783,
            0.003060588697136,
            0,
            -0.004574266843525,
            -0.005527530217305,
            0,
            0.007970838353584,
            0.009473448478316,
            0,
            -0.013190283159248,
            -0.015476487213699,
            0,
            0.021063800568830,
            0.024531612691963,
            0,
            -0.033469800335079,
            -0.039266345965824,
            0,
            0.055115000503041,
            0.066479258736773,
            0,
            -0.103488680132006,
            -0.136913709817207,
            0,
            0.316296179555473,
            0.704251614946481,
            1.000000000000000,
            1.086362355697893,
            0.958594196083290,
            0.707106781186548,
            0.439365947446575,
            0.208264996080477,
            0.000000000000000,
            -0.222591316028712,
            -0.469073222402289,
            -0.707106781186547,
            -0.888770206983388,
            -0.985453057043050,
            -1.000000000000000,
            -0.951667378340539,
            -0.853904136800703,
            -0.707106781186548,
            -0.508682029357705,
            -0.266106706593376,
            -0.000000000000000,
            0.263864387323959,
            0.504160965552528,
            0.707106781186547,
            0.863324002566594,
            0.963802028619039,
        ]
    )

    fir = sdr.Interpolator(3, streaming=True)
    y = fir(x)

    assert fir.interpolation == 3
    assert fir.decimation == 1
    assert fir.rate == 3
    # debug_plot(y, y_truth)
    np.testing.assert_almost_equal(y, y_truth)


def test_3_linear():
    """
    MATLAB:
        >> x = cos(pi/4*(0:19)');
        >> fir = dsp.Interpolator(3, 'Linear');
        >> y = fir(x);
    """
    x = np.cos(np.pi / 4 * np.arange(20))
    y_truth = np.array(
        [
            0.333333333333333,
            0.666666666666667,
            1.000000000000000,
            0.902368927062182,
            0.804737854124365,
            0.707106781186548,
            0.471404520791032,
            0.235702260395516,
            0.000000000000000,
            -0.235702260395516,
            -0.471404520791032,
            -0.707106781186547,
            -0.804737854124365,
            -0.902368927062182,
            -1.000000000000000,
            -0.902368927062183,
            -0.804737854124365,
            -0.707106781186548,
            -0.471404520791032,
            -0.235702260395516,
            -0.000000000000000,
            0.235702260395516,
            0.471404520791032,
            0.707106781186547,
            0.804737854124365,
            0.902368927062182,
            1.000000000000000,
            0.902368927062183,
            0.804737854124365,
            0.707106781186548,
            0.471404520791032,
            0.235702260395516,
            0.000000000000000,
            -0.235702260395515,
            -0.471404520791031,
            -0.707106781186547,
            -0.804737854124364,
            -0.902368927062182,
            -1.000000000000000,
            -0.902368927062182,
            -0.804737854124365,
            -0.707106781186547,
            -0.471404520791032,
            -0.235702260395516,
            -0.000000000000000,
            0.235702260395515,
            0.471404520791031,
            0.707106781186547,
            0.804737854124364,
            0.902368927062182,
            1.000000000000000,
            0.902368927062182,
            0.804737854124365,
            0.707106781186547,
            0.471404520791032,
            0.235702260395516,
            0.000000000000001,
            -0.235702260395515,
            -0.471404520791031,
            -0.707106781186546,
        ]
    )

    fir = sdr.Interpolator(3, "linear-matlab", streaming=True)
    y = fir(x)

    assert fir.interpolation == 3
    assert fir.decimation == 1
    assert fir.rate == 3
    # debug_plot(y, y_truth)
    np.testing.assert_almost_equal(y, y_truth)


def test_3_zoh():
    """
    MATLAB:
        >> x = cos(pi/4*(0:19)');
        >> fir = dsp.Interpolator(3, 'ZOH');
        >> y = fir(x);
    """
    x = np.cos(np.pi / 4 * np.arange(20))
    y_truth = np.array(
        [
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            0.707106781186548,
            0.707106781186548,
            0.707106781186548,
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            -0.707106781186547,
            -0.707106781186547,
            -0.707106781186547,
            -1.000000000000000,
            -1.000000000000000,
            -1.000000000000000,
            -0.707106781186548,
            -0.707106781186548,
            -0.707106781186548,
            -0.000000000000000,
            -0.000000000000000,
            -0.000000000000000,
            0.707106781186547,
            0.707106781186547,
            0.707106781186547,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            0.707106781186548,
            0.707106781186548,
            0.707106781186548,
            0.000000000000000,
            0.000000000000000,
            0.000000000000000,
            -0.707106781186547,
            -0.707106781186547,
            -0.707106781186547,
            -1.000000000000000,
            -1.000000000000000,
            -1.000000000000000,
            -0.707106781186547,
            -0.707106781186547,
            -0.707106781186547,
            -0.000000000000000,
            -0.000000000000000,
            -0.000000000000000,
            0.707106781186547,
            0.707106781186547,
            0.707106781186547,
            1.000000000000000,
            1.000000000000000,
            1.000000000000000,
            0.707106781186547,
            0.707106781186547,
            0.707106781186547,
            0.000000000000001,
            0.000000000000001,
            0.000000000000001,
            -0.707106781186546,
            -0.707106781186546,
            -0.707106781186546,
        ]
    )

    fir = sdr.Interpolator(3, "zoh", streaming=True)
    y = fir(x)

    assert fir.interpolation == 3
    assert fir.decimation == 1
    assert fir.rate == 3
    # debug_plot(y, y_truth)
    np.testing.assert_almost_equal(y, y_truth)


def test_srrc_0p5_6():
    """
    MATLAB:
        >> sps = 4;
        >> h = rcosdesign(0.5, 6, sps);
        >> s = randi([0 3], 10, 1);
        >> x = pskmod(s, 4);
        >> fir = dsp.Interpolator(sps, h);
        >> y = fir(x);
    """
    h = np.array(
        [
            0.001516034308196,
            -0.008227494944468,
            -0.007503986978858,
            0.007735482835249,
            0.021224480314744,
            0.007735482835249,
            -0.037519934894290,
            -0.078435451803924,
            -0.053061200786861,
            0.078435451803924,
            0.289368332423475,
            0.487335418620697,
            0.568412222595146,
            0.487335418620697,
            0.289368332423475,
            0.078435451803924,
            -0.053061200786861,
            -0.078435451803924,
            -0.037519934894290,
            0.007735482835249,
            0.021224480314744,
            0.007735482835249,
            -0.007503986978858,
            -0.008227494944468,
            0.001516034308196,
        ]
    )
    x = np.array(
        [
            -0.000000000000000 - 1.000000000000000j,
            -0.000000000000000 - 1.000000000000000j,
            -1.000000000000000 + 0.000000000000000j,
            -0.000000000000000 - 1.000000000000000j,
            -1.000000000000000 + 0.000000000000000j,
            0.000000000000000 + 1.000000000000000j,
            0.000000000000000 + 1.000000000000000j,
            -1.000000000000000 + 0.000000000000000j,
            1.000000000000000 + 0.000000000000000j,
            1.000000000000000 + 0.000000000000000j,
        ]
    )
    y_truth = np.array(
        [
            -0.000000000000000 - 0.001516034308196j,
            0.000000000000000 + 0.008227494944468j,
            0.000000000000000 + 0.007503986978858j,
            -0.000000000000000 - 0.007735482835249j,
            -0.000000000000000 - 0.022740514622940j,
            0.000000000000000 + 0.000492012109218j,
            0.000000000000000 + 0.045023921873148j,
            0.000000000000000 + 0.070699968968675j,
            -0.001516034308196 + 0.031836720472116j,
            0.008227494944468 - 0.086170934639173j,
            0.007503986978858 - 0.251848397529184j,
            -0.007735482835249 - 0.408899966816774j,
            -0.021224480314744 - 0.516867056116481j,
            -0.007735482835249 - 0.557543375480154j,
            0.037519934894290 - 0.571232677868091j,
            0.078435451803924 - 0.573506353259870j,
            0.051545166478665 - 0.536575502123030j,
            -0.070207956859456 - 0.416635449652023j,
            -0.281864345444617 - 0.214328462634894j,
            -0.495070901455947 - 0.007735482835249j,
            -0.589636702909890 + 0.086413955567173j,
            -0.495070901455947 - 0.015962977779717j,
            -0.251848397529184 - 0.251848397529184j,
            -0.000000000000000 - 0.479107923676230j,
            0.106122401573721 - 0.568412222595146j,
            -0.000000000000000 - 0.495562913565165j,
            -0.251848397529184 - 0.326888267317765j,
            -0.495070901455947 - 0.140907925828131j,
            -0.591152737218086 + 0.019708446006548j,
            -0.486843406511479 + 0.164606386443097j,
            -0.274360358465759 + 0.289368332423475j,
            -0.077943439694705 + 0.401164483981524j,
            0.031836720472116 + 0.494126541493541j,
            0.062472474024207 + 0.558035387589372j,
            0.067535882809722 + 0.586240651825807j,
            0.078435451803924 + 0.573998365369089j,
            0.054577235095057 + 0.513834987500089j,
            -0.086662946748391 + 0.408899966816774j,
            -0.326888267317765 + 0.251848397529184j,
            -0.549807892644904 + 0.086170934639173j,
        ]
    )

    fir = sdr.Interpolator(4, h, streaming=True)
    y = fir(x)

    assert fir.interpolation == 4
    assert fir.decimation == 1
    assert fir.rate == 4
    # debug_plot(y, y_truth)
    np.testing.assert_almost_equal(y, y_truth)


def test_srrc_0p9_4():
    """
    MATLAB:
        >> sps = 5;
        >> h = rcosdesign(0.9, 4, sps);
        >> s = randi([0 3], 10, 1);
        >> x = pskmod(s, 4);
        >> fir = dsp.Interpolator(sps, h);
        >> y = fir(x);
    """
    h = np.array(
        [
            -0.008979177692249,
            0.002075680859996,
            0.014470828983618,
            0.008344582812409,
            -0.020970280947284,
            -0.044436166600014,
            -0.010479514202742,
            0.114388398953594,
            0.306463994703480,
            0.484592171235420,
            0.557274483307239,
            0.484592171235420,
            0.306463994703480,
            0.114388398953594,
            -0.010479514202742,
            -0.044436166600014,
            -0.020970280947284,
            0.008344582812409,
            0.014470828983618,
            0.002075680859996,
            -0.008979177692249,
        ]
    )
    x = np.array(
        [
            -1.000000000000000 + 0.000000000000000j,
            -1.000000000000000 + 0.000000000000000j,
            1.000000000000000 + 0.000000000000000j,
            -0.000000000000000 - 1.000000000000000j,
            0.000000000000000 + 1.000000000000000j,
            0.000000000000000 + 1.000000000000000j,
            -0.000000000000000 - 1.000000000000000j,
            -1.000000000000000 + 0.000000000000000j,
            0.000000000000000 + 1.000000000000000j,
            -1.000000000000000 + 0.000000000000000j,
        ]
    )
    y_truth = np.array(
        [
            0.008979177692249 - 0.000000000000000j,
            -0.002075680859996 + 0.000000000000000j,
            -0.014470828983618 + 0.000000000000000j,
            -0.008344582812409 + 0.000000000000000j,
            0.020970280947284 - 0.000000000000000j,
            0.053415344292262 - 0.000000000000000j,
            0.008403833342745 - 0.000000000000000j,
            -0.128859227937211 + 0.000000000000000j,
            -0.314808577515888 + 0.000000000000000j,
            -0.463621890288136 + 0.000000000000000j,
            -0.521817494399474 + 0.000000000000000j,
            -0.472036976172682 + 0.000000000000000j,
            -0.406381564673456 + 0.000000000000000j,
            -0.412507810844664 + 0.000000000000000j,
            -0.495082937979962 + 0.000000000000000j,
            -0.557274483307239 + 0.008979177692249j,
            -0.474101404490878 - 0.002075680859996j,
            -0.200420178562295 - 0.014470828983618j,
            0.177604766766268 - 0.008344582812409j,
            0.492996004578166 + 0.020970280947284j,
            0.610689827599501 + 0.035456988907765j,
            0.505562452182704 + 0.012555195062738j,
            0.298119411891071 - 0.099917569969976j,
            0.099917569969976 - 0.298119411891071j,
            -0.012555195062738 - 0.505562452182704j,
            -0.035456988907765 - 0.610689827599501j,
            -0.020970280947284 - 0.492996004578166j,
            0.008344582812409 - 0.177604766766268j,
            0.014470828983618 + 0.200420178562295j,
            0.002075680859996 + 0.474101404490878j,
            -0.008979177692249 + 0.566253660999487j,
            0.000000000000000 + 0.493007257119966j,
            0.000000000000000 + 0.398036981861047j,
            0.000000000000000 + 0.398036981861047j,
            0.000000000000000 + 0.493007257119966j,
            0.008979177692249 + 0.566253660999487j,
            -0.002075680859996 + 0.474101404490878j,
            -0.014470828983618 + 0.200420178562295j,
            -0.008344582812409 - 0.177604766766268j,
            0.020970280947284 - 0.492996004578166j,
            0.044436166600014 - 0.619669005291750j,
            0.010479514202742 - 0.503486771322708j,
            -0.114388398953594 - 0.283648582907453j,
            -0.306463994703480 - 0.091572987157567j,
            -0.484592171235420 - 0.008415085884546j,
            -0.548295305614990 - 0.008979177692249j,
            -0.486667852095416 + 0.010490766744542j,
            -0.320934823687097 + 0.106043816141185j,
            -0.122732981766002 + 0.291993165719862j,
            0.031449795150026 + 0.482516490375424j,
        ]
    )

    fir = sdr.Interpolator(5, h, streaming=True)
    y = fir(x)

    assert fir.interpolation == 5
    assert fir.decimation == 1
    assert fir.rate == 5
    # debug_plot(y, y_truth)
    np.testing.assert_almost_equal(y, y_truth)


def test_srrc_0p1_7():
    """
    MATLAB:
        >> sps = 6;
        >> h = rcosdesign(0.1, 7, sps);
        >> s = randi([0 3], 10, 1);
        >> x = pskmod(s, 4);
        >> fir = dsp.Interpolator(sps, h);
        >> y = fir(x);
    """
    h = np.array(
        [
            -0.030809532987194,
            -0.033550304159821,
            -0.026427564389386,
            -0.010247568208595,
            0.011392294086020,
            0.032777825557765,
            0.047435629159881,
            0.049876901519861,
            0.037296749981352,
            0.010769192077375,
            -0.024431323957485,
            -0.059556113491369,
            -0.084075461797583,
            -0.088005789079879,
            -0.064353293177153,
            -0.011091111139443,
            0.067836715588281,
            0.162785749175542,
            0.259968653076483,
            0.344008641127492,
            0.400953817067850,
            0.421094223786079,
            0.400953817067850,
            0.344008641127492,
            0.259968653076483,
            0.162785749175542,
            0.067836715588281,
            -0.011091111139443,
            -0.064353293177153,
            -0.088005789079879,
            -0.084075461797583,
            -0.059556113491369,
            -0.024431323957485,
            0.010769192077375,
            0.037296749981352,
            0.049876901519861,
            0.047435629159881,
            0.032777825557765,
            0.011392294086020,
            -0.010247568208595,
            -0.026427564389386,
            -0.033550304159821,
            -0.030809532987194,
        ]
    )
    x = np.array(
        [
            1.000000000000000 + 0.000000000000000j,
            0.000000000000000 + 1.000000000000000j,
            -1.000000000000000 + 0.000000000000000j,
            -1.000000000000000 + 0.000000000000000j,
            1.000000000000000 + 0.000000000000000j,
            0.000000000000000 + 1.000000000000000j,
            -0.000000000000000 - 1.000000000000000j,
            0.000000000000000 + 1.000000000000000j,
            -0.000000000000000 - 1.000000000000000j,
            -0.000000000000000 - 1.000000000000000j,
        ]
    )
    y_truth = np.array(
        [
            -0.030809532987194 + 0.000000000000000j,
            -0.033550304159821 + 0.000000000000000j,
            -0.026427564389386 + 0.000000000000000j,
            -0.010247568208595 + 0.000000000000000j,
            0.011392294086020 + 0.000000000000000j,
            0.032777825557765 + 0.000000000000000j,
            0.047435629159881 - 0.030809532987194j,
            0.049876901519861 - 0.033550304159821j,
            0.037296749981352 - 0.026427564389386j,
            0.010769192077375 - 0.010247568208595j,
            -0.024431323957485 + 0.011392294086020j,
            -0.059556113491369 + 0.032777825557765j,
            -0.053265928810389 + 0.047435629159881j,
            -0.054455484920057 + 0.049876901519861j,
            -0.037925728787767 + 0.037296749981352j,
            -0.000843542930848 + 0.010769192077375j,
            0.056444421502261 - 0.024431323957485j,
            0.130007923617777 - 0.059556113491369j,
            0.243342556903797 - 0.084075461797583j,
            0.327682043767453 - 0.088005789079879j,
            0.390084631475885 - 0.064353293177153j,
            0.420572599917300 - 0.011091111139443j,
            0.413992846939316 + 0.067836715588281j,
            0.370786929061097 + 0.162785749175542j,
            0.265798952726991 + 0.259968653076483j,
            0.167364332575738 + 0.344008641127492j,
            0.068465694394696 + 0.400953817067850j,
            -0.021016760285969 + 0.421094223786079j,
            -0.096366390721928 + 0.400953817067850j,
            -0.158457599206286 + 0.344008641127492j,
            -0.212533023916602 + 0.229159120089289j,
            -0.265682064019122 + 0.129235445015721j,
            -0.323735097866831 + 0.041409151198895j,
            -0.388464728491887 - 0.021338679348038j,
            -0.455925106632265 - 0.052960999091133j,
            -0.516473602274542 - 0.055227963522114j,
            -0.556577138790669 - 0.005830299650508j,
            -0.562022353825148 + 0.023871092188313j,
            -0.521751531747264 + 0.039292990413253j,
            -0.431341791994674 + 0.031785952363344j,
            -0.295191372691803 + 0.001473131937847j,
            -0.126767407031893 - 0.042457037529273j,
            0.053265928810389 - 0.114884994784777j,
            0.240779005443320 - 0.138655169201796j,
            0.357548425437055 - 0.116685313461870j,
            0.421416142848148 - 0.042355439634007j,
            0.428010360263651 + 0.077232769242400j,
            0.382137528687510 + 0.221569384064855j,
            0.296608485714185 + 0.391479744033947j,
            0.189564037109146 + 0.515441635887053j,
            0.080875745459746 + 0.529031424615741j,
            -0.011612735008223 + 0.453202095211492j,
            -0.075222478769118 + 0.297293483436065j,
            -0.104332386439918 + 0.088888952902816j,
            -0.100701557970270 - 0.100701557970270j,
            -0.092333939049134 - 0.285555278391869j,
            -0.035823618043505 - 0.408339580248688j,
            0.021016760285969 - 0.443798069933745j,
            0.063724314370738 - 0.384431364785257j,
            0.083427205679682 - 0.242450393098225j,
        ]
    )

    fir = sdr.Interpolator(6, h, streaming=True)
    y = fir(x)

    assert fir.interpolation == 6
    assert fir.decimation == 1
    assert fir.rate == 6
    # debug_plot(y, y_truth)
    np.testing.assert_almost_equal(y, y_truth)
