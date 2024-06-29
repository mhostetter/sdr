import numpy as np

import sdr


def test_bpsk_rect():
    rng = np.random.default_rng()
    phi = rng.uniform(0, 360)
    samples_per_symbol = 10
    h = sdr.rectangular(samples_per_symbol)
    psk = sdr.PSK(2, phase_offset=phi, samples_per_symbol=samples_per_symbol, pulse_shape=h)
    _verify_demodulation(psk)


def test_bpsk_srrc():
    rng = np.random.default_rng()
    phi = rng.uniform(0, 360)
    samples_per_symbol = 10
    h = sdr.root_raised_cosine(0.5, 16, samples_per_symbol)
    psk = sdr.PSK(2, phase_offset=phi, samples_per_symbol=samples_per_symbol, pulse_shape=h)
    _verify_demodulation(psk)


def test_qpsk_rect():
    rng = np.random.default_rng()
    phi = rng.uniform(0, 360)
    samples_per_symbol = 10
    h = sdr.rectangular(samples_per_symbol)
    psk = sdr.PSK(4, phase_offset=phi, samples_per_symbol=samples_per_symbol, pulse_shape=h)
    _verify_demodulation(psk)


def test_qpsk_srrc():
    rng = np.random.default_rng()
    phi = rng.uniform(0, 360)
    samples_per_symbol = 10
    h = sdr.root_raised_cosine(0.5, 16, samples_per_symbol)
    psk = sdr.PSK(4, phase_offset=phi, samples_per_symbol=samples_per_symbol, pulse_shape=h)
    _verify_demodulation(psk)


def test_8psk_rect():
    rng = np.random.default_rng()
    phi = rng.uniform(0, 360)
    samples_per_symbol = 10
    h = sdr.rectangular(samples_per_symbol)
    psk = sdr.PSK(8, phase_offset=phi, samples_per_symbol=samples_per_symbol, pulse_shape=h)
    _verify_demodulation(psk)


def test_8psk_srrc():
    rng = np.random.default_rng()
    phi = rng.uniform(0, 360)
    samples_per_symbol = 10
    h = sdr.root_raised_cosine(0.5, 16, samples_per_symbol)
    psk = sdr.PSK(8, phase_offset=phi, samples_per_symbol=samples_per_symbol, pulse_shape=h)
    _verify_demodulation(psk)


def _verify_demodulation(psk: sdr.PSK):
    rng = np.random.default_rng()
    s = rng.integers(0, psk.order, 20)
    a = psk.map_symbols(s)
    x = psk.modulate(s)
    s_hat, a_tilde, a_hat = psk.demodulate(x)

    assert np.array_equal(s, s_hat)
    np.testing.assert_array_almost_equal(a, a_hat, decimal=3)
