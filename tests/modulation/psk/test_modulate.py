import numpy as np

import sdr


def test_bpsk_rect():
    phi = np.random.uniform(0, 360)
    sps = 10
    h = sdr.rectangular(sps, norm="power")
    psk = sdr.PSK(2, phase_offset=phi, sps=sps, pulse_shape=h)
    _verify_modulation(psk)


def test_bpsk_rc():
    phi = np.random.uniform(0, 360)
    sps = 10
    h = sdr.raised_cosine(0.5, 16, sps, norm="power")
    psk = sdr.PSK(2, phase_offset=phi, sps=sps, pulse_shape=h)
    _verify_modulation(psk)


def test_qpsk_rect():
    phi = np.random.uniform(0, 360)
    sps = 10
    h = sdr.rectangular(sps, norm="power")
    psk = sdr.PSK(4, phase_offset=phi, sps=sps, pulse_shape=h)
    _verify_modulation(psk)


def test_qpsk_rc():
    phi = np.random.uniform(0, 360)
    sps = 10
    h = sdr.raised_cosine(0.5, 16, sps, norm="power")
    psk = sdr.PSK(4, phase_offset=phi, sps=sps, pulse_shape=h)
    _verify_modulation(psk)


def test_8psk_rect():
    phi = np.random.uniform(0, 360)
    sps = 10
    h = sdr.rectangular(sps, norm="power")
    psk = sdr.PSK(8, phase_offset=phi, sps=sps, pulse_shape=h)
    _verify_modulation(psk)


def test_8psk_rc():
    phi = np.random.uniform(0, 360)
    sps = 10
    h = sdr.raised_cosine(0.5, 16, sps, norm="power")
    psk = sdr.PSK(8, phase_offset=phi, sps=sps, pulse_shape=h)
    _verify_modulation(psk)


def _verify_modulation(psk: sdr.PSK):
    s = np.random.randint(0, psk.order, 20)
    a = psk.map_symbols(s)
    x = psk.modulate(s)

    offset = psk.pulse_shape.size // 2
    a_hat = x[offset : offset + s.size * psk.sps : psk.sps]

    # import matplotlib.pyplot as plt

    # plt.figure()
    # sdr.plot.constellation(a)
    # sdr.plot.constellation(a_hat)

    # plt.figure()
    # sdr.plot.time_domain(a)
    # sdr.plot.time_domain(a_hat)
    # plt.show()

    np.testing.assert_array_almost_equal(a, a_hat, decimal=3)
