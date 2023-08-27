"""
Matlab:
    x_hat = exp(1i * 2 * pi * rand(1, 20));
    disp(transpose(x_hat))
    for M = [2 4 8 16]
        for map = ["bin", "gray"]
            disp(M)
            disp(map)
            phi = 2*pi*(rand - 0.5);
            s = pskdemod(x_hat, M, phi, map);
            disp(phi)
            disp(s')
        end
    end
"""
import numpy as np

import sdr

X_HAT = np.array(
    [
        -0.906775305398710 - 0.421614214085668j,
        0.007724076072342 + 0.999970168879466j,
        -0.421804615671564 + 0.906686751970141j,
        0.847018062445188 - 0.531564108919705j,
        0.751024654669571 - 0.660274161298511j,
        0.276512887733222 - 0.961010209580228j,
        0.893304960564346 - 0.449451051207062j,
        0.432363030358946 + 0.901699622922639j,
        -0.993932551307667 - 0.109991288068795j,
        -0.698141503472522 - 0.715959804129482j,
        0.854872432413258 - 0.518838244831508j,
        -0.514610571827507 - 0.857424025417626j,
        -0.767270560521734 + 0.641323543117407j,
        -0.062742978013181 - 0.998029718350129j,
        0.412286487015338 - 0.911054253392492j,
        -0.807741719267517 - 0.589536525547616j,
        0.860751415756623 + 0.509025539902439j,
        0.883347788854531 - 0.468718128437348j,
        0.943828559442702 + 0.330435546484203j,
        -0.985618112940125 - 0.168987974259550j,
    ]
)


def test_bpsk_bin():
    phi = np.rad2deg(-2.394815492222444)
    psk = sdr.PSK(2, phase_offset=phi, symbol_labels="bin")
    s = psk.decide_symbols(X_HAT)
    s_truth = np.array([0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    assert np.array_equal(s, s_truth)


def test_bpsk_gray():
    phi = np.rad2deg(-0.753083802680199)
    psk = sdr.PSK(2, phase_offset=phi, symbol_labels="gray")
    s = psk.decide_symbols(X_HAT)
    s_truth = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
    assert np.array_equal(s, s_truth)


def test_qpsk_bin():
    phi = np.rad2deg(1.965584775048307)
    psk = sdr.PSK(4, phase_offset=phi, symbol_labels="bin")
    s = psk.decide_symbols(X_HAT)
    s_truth = np.array([1, 0, 0, 2, 2, 2, 2, 3, 1, 1, 2, 1, 0, 2, 2, 1, 3, 2, 3, 1])
    assert np.array_equal(s, s_truth)


def test_qpsk_gray():
    phi = np.rad2deg(-1.607892783327202)
    psk = sdr.PSK(4, phase_offset=phi, symbol_labels="gray")
    s = psk.decide_symbols(X_HAT)
    s_truth = np.array([2, 3, 3, 1, 1, 0, 1, 3, 2, 0, 1, 0, 2, 0, 0, 2, 1, 1, 1, 2])
    assert np.array_equal(s, s_truth)


def test_8psk_bin():
    phi = np.rad2deg(2.415398803363657)
    psk = sdr.PSK(8, phase_offset=phi, symbol_labels="bin")
    s = psk.decide_symbols(X_HAT)
    s_truth = np.array([1, 7, 7, 4, 4, 3, 4, 6, 1, 2, 4, 2, 0, 3, 3, 2, 6, 4, 5, 1])
    assert np.array_equal(s, s_truth)


def test_8psk_gray():
    phi = np.rad2deg(1.336099216449522)
    psk = sdr.PSK(8, phase_offset=phi, symbol_labels="gray")
    s = psk.decide_symbols(X_HAT)
    s_truth = np.array([2, 0, 1, 5, 7, 7, 5, 0, 3, 2, 5, 6, 1, 6, 7, 2, 4, 5, 4, 2])
    assert np.array_equal(s, s_truth)


def test_16psk_bin():
    phi = np.rad2deg(-0.765615943499161)
    psk = sdr.PSK(16, phase_offset=phi, symbol_labels="bin")
    s = psk.decide_symbols(X_HAT)
    s_truth = np.array([11, 6, 7, 1, 0, 15, 1, 5, 10, 12, 1, 13, 8, 14, 15, 12, 3, 1, 3, 10])
    assert np.array_equal(s, s_truth)


def test_16psk_gray():
    phi = np.rad2deg(-1.577584611111288)
    psk = sdr.PSK(16, phase_offset=phi, symbol_labels="gray")
    s = psk.decide_symbols(X_HAT)
    s_truth = np.array([11, 12, 13, 2, 3, 1, 2, 4, 10, 9, 2, 8, 15, 0, 1, 9, 7, 2, 7, 10])
    assert np.array_equal(s, s_truth)
