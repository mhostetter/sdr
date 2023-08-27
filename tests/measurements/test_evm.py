"""
Matlab:
    symbol_map = pskmod(0:3, 4, pi/4, "gray");
    s = randi([0 3], 50, 1);
    x = pskmod(s, 4, pi/4, "gray");
    x_hat = awgn(x, 20);
    disp(s)
    disp(x_hat)
    evm = comm.EVM( ...
        Normalization="Average reference signal power" ...
    );
    evm(x, x_hat)'
    evm = comm.EVM( ...
        Normalization="Average constellation power", ...
        AverageConstellationPower=mean(abs(x_hat).^2) ...
    );
    evm(x, x_hat)'
    evm = comm.EVM( ...
        Normalization="Peak constellation power", ...
        PeakConstellationPower=max(abs(x_hat).^2) ...
    );
    evm(x, x_hat)'
    evm = comm.EVM( ...
        XPercentileEVMOutputPort=true, ...
        XPercentileValue=10 ...
    );
    [rms perc] = evm(x, x_hat);
    perc'
    evm = comm.EVM( ...
        XPercentileEVMOutputPort=true, ...
        XPercentileValue=50 ...
    );
    [rms perc] = evm(x, x_hat);
    perc'
    evm = comm.EVM( ...
        XPercentileEVMOutputPort=true, ...
        XPercentileValue=90 ...
    );
    [rms perc] = evm(x, x_hat);
    perc'
"""
import numpy as np
import pytest

import sdr

S = np.array(
    [
        2,
        1,
        0,
        2,
        1,
        3,
        1,
        2,
        2,
        2,
        2,
        1,
        0,
        2,
        0,
        3,
        1,
        3,
        3,
        1,
        0,
        3,
        2,
        2,
        2,
        1,
        2,
        1,
        3,
        1,
        3,
        3,
        0,
        1,
        3,
        1,
        1,
        2,
        3,
        3,
        2,
        2,
        2,
        1,
        3,
        1,
        3,
        3,
        0,
        0,
    ]
)

X_HAT = np.array(
    [
        0.620097036355970 - 0.757968684865226j,
        -0.700313753869277 + 0.642262812552823j,
        0.546047381655515 + 0.756527316929334j,
        0.698759834441931 - 0.708936601061648j,
        -0.632283951683652 + 0.730303443410090j,
        -0.676256069553356 - 0.756679023556570j,
        -0.607296985431214 + 0.553923515203542j,
        0.706073544650952 - 0.703587280355626j,
        0.669079183953607 - 0.662833681583229j,
        0.577369588954867 - 0.626243800330979j,
        0.893467134858904 - 0.835917383494016j,
        -0.686545962796005 + 0.728099722869958j,
        0.782455983953758 + 0.718315407258294j,
        0.744953065286946 - 0.703843505280321j,
        0.679717429383400 + 0.699285781186680j,
        -0.729708685959369 - 0.569599005615022j,
        -0.885617521088574 + 0.625225585442871j,
        -0.778261794579096 - 0.692866315342902j,
        -0.737099215742626 - 0.675867679596884j,
        -0.753861184359404 + 0.567331659037931j,
        0.758553560115261 + 0.738135091471210j,
        -0.674819282884428 - 0.604008662230154j,
        0.671315107871070 - 0.727389018277134j,
        0.623339688366707 - 0.727702824662731j,
        0.762387503759289 - 0.810076428543931j,
        -0.712790697998218 + 0.728535833374632j,
        0.730735632035205 - 0.633811327494550j,
        -0.682638064238007 + 0.618193923854139j,
        -0.715556129756411 - 0.802667494230882j,
        -0.656463453721405 + 0.664650675717629j,
        -0.690988704378097 - 0.700314233525801j,
        -0.703807672691549 - 0.627858342758267j,
        0.631817832584889 + 0.746850193943219j,
        -0.704288255418999 + 0.701290390238423j,
        -0.642274702034144 - 0.766756755838617j,
        -0.736010330503131 + 0.711412998318157j,
        -0.678746382812940 + 0.745210422816902j,
        0.752006463675763 - 0.758144666171453j,
        -0.471598857343963 - 0.690843171513840j,
        -0.757909423736994 - 0.688005936633361j,
        0.692684157132057 - 0.613301940653193j,
        0.676707389320649 - 0.597071726820518j,
        0.598709461629948 - 0.838857257060654j,
        -0.694117380897259 + 0.784496608092530j,
        -0.684096265363600 - 0.753032243796739j,
        -0.732736317723687 + 0.614306497276497j,
        -0.710285441527914 - 0.749984510793444j,
        -0.714346155567909 - 0.740860768573361j,
        0.685737485567278 + 0.729198233018627j,
        0.646748509168238 + 0.691850347770370j,
    ]
)


def test_exceptions():
    psk = sdr.PSK(4, phase_offset=45)
    with pytest.raises(ValueError):
        evm = sdr.evm(X_HAT, psk.symbol_map, norm="invalid")
    with pytest.raises(ValueError):
        evm = sdr.evm(X_HAT, psk.symbol_map, output="invalid")
    with pytest.raises(ValueError):
        # Output percentile must be in [0, 100]
        evm = sdr.evm(X_HAT, psk.symbol_map, output=-1)
    with pytest.raises(ValueError):
        # Output percentile must be in [0, 100]
        evm = sdr.evm(X_HAT, psk.symbol_map, output=101)


def test_average_power_ref():
    psk = sdr.PSK(4, phase_offset=45)
    evm = sdr.evm(X_HAT, psk.symbol_map, norm="average-power-ref")
    assert evm == pytest.approx(9.913846806617736)

    x = psk.map_symbols(S)
    evm = sdr.evm(X_HAT, x, norm="average-power-ref")
    assert evm == pytest.approx(9.913846806617736)


def test_average_power():
    psk = sdr.PSK(4, phase_offset=45)
    evm = sdr.evm(X_HAT, psk.symbol_map, norm="average-power")
    assert evm == pytest.approx(10.005285125256888)

    x = psk.map_symbols(S)
    evm = sdr.evm(X_HAT, x, norm="average-power")
    assert evm == pytest.approx(10.005285125256888)


def test_peak_power():
    psk = sdr.PSK(4, phase_offset=45)
    evm = sdr.evm(X_HAT, psk.symbol_map, norm="peak-power")
    assert evm == pytest.approx(8.102616784925608)

    x = psk.map_symbols(S)
    evm = sdr.evm(X_HAT, x, norm="peak-power")
    assert evm == pytest.approx(8.102616784925608)


# TODO: For some reason Matlab computes percentiles differently??

# def test_10th_percentile():
#     psk = sdr.PSK(4, phase_offset=45)
#     evm = sdr.evm(X_HAT, psk.symbol_map, output=10)
#     assert evm == pytest.approx(2.835000000000000)


# def test_50th_percentile():
#     psk = sdr.PSK(4, phase_offset=45)
#     evm = sdr.evm(X_HAT, psk.symbol_map, output=50)
#     assert evm == pytest.approx(7.605000000000000)


# def test_90th_percentile():
#     psk = sdr.PSK(4, phase_offset=45)
#     evm = sdr.evm(X_HAT, psk.symbol_map, output=90)
#     assert evm == pytest.approx(17.055000000000000)
