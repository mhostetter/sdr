import numpy as np
import pytest

import sdr


def test_bpsk_symbol_map_bin():
    """
    Matlab:
        >> M = 2; transpose(pskmod(0:M-1, M, 0, 'bin'))
    """
    psk = sdr.PSK(2, symbol_labels="bin")
    symbol_map = np.array(
        [
            1.0000 + 0.0000j,
            -1.0000 + 0.0000j,
        ]
    )
    np.testing.assert_almost_equal(psk.symbol_map, symbol_map, decimal=4)


def test_bpsk_symbol_map_gray():
    """
    Matlab:
        >> M = 2; transpose(pskmod(0:M-1, M, 0, 'gray'))
    """
    psk = sdr.PSK(2, symbol_labels="gray")
    symbol_map = np.array(
        [
            1.0000 + 0.0000j,
            -1.0000 + 0.0000j,
        ]
    )
    np.testing.assert_almost_equal(psk.symbol_map, symbol_map, decimal=4)


def test_qpsk_symbol_map_bin():
    """
    Matlab:
        >> M = 4; transpose(pskmod(0:M-1, M, 0, 'bin'))
    """
    psk = sdr.PSK(4, symbol_labels="bin")
    symbol_map = np.array(
        [
            1.0000 + 0.0000j,
            0.0000 + 1.0000j,
            -1.0000 + 0.0000j,
            -0.0000 - 1.0000j,
        ]
    )
    np.testing.assert_almost_equal(psk.symbol_map, symbol_map, decimal=4)


def test_qpsk_symbol_map_gray():
    """
    Matlab:
        >> M = 4; transpose(pskmod(0:M-1, M, 0, 'gray'))
    """
    psk = sdr.PSK(4, symbol_labels="gray")
    symbol_map = np.array(
        [
            1.0000 + 0.0000j,
            0.0000 + 1.0000j,
            -0.0000 - 1.0000j,
            -1.0000 + 0.0000j,
        ]
    )
    np.testing.assert_almost_equal(psk.symbol_map, symbol_map, decimal=4)


def test_8psk_symbol_map_bin():
    """
    Matlab:
        >> M = 8; transpose(pskmod(0:M-1, M, 0, 'bin'))
    """
    psk = sdr.PSK(8, symbol_labels="bin")
    symbol_map = np.array(
        [
            1.0000 + 0.0000j,
            0.7071 + 0.7071j,
            0.0000 + 1.0000j,
            -0.7071 + 0.7071j,
            -1.0000 + 0.0000j,
            -0.7071 - 0.7071j,
            -0.0000 - 1.0000j,
            0.7071 - 0.7071j,
        ]
    )
    np.testing.assert_almost_equal(psk.symbol_map, symbol_map, decimal=4)


def test_8psk_symbol_map_gray():
    """
    Matlab:
        >> M = 8; transpose(pskmod(0:M-1, M, 0, 'gray'))
    """
    psk = sdr.PSK(8, symbol_labels="gray")
    symbol_map = np.array(
        [
            1.0000 + 0.0000j,
            0.7071 + 0.7071j,
            -0.7071 + 0.7071j,
            0.0000 + 1.0000j,
            0.7071 - 0.7071j,
            -0.0000 - 1.0000j,
            -1.0000 + 0.0000j,
            -0.7071 - 0.7071j,
        ]
    )
    np.testing.assert_almost_equal(psk.symbol_map, symbol_map, decimal=4)


def test_16psk_symbol_map_bin():
    """
    Matlab:
        >> M = 16; transpose(pskmod(0:M-1, M, 0, 'bin'))
    """
    psk = sdr.PSK(16, symbol_labels="bin")
    symbol_map = np.array(
        [
            1.0000 + 0.0000j,
            0.9239 + 0.3827j,
            0.7071 + 0.7071j,
            0.3827 + 0.9239j,
            0.0000 + 1.0000j,
            -0.3827 + 0.9239j,
            -0.7071 + 0.7071j,
            -0.9239 + 0.3827j,
            -1.0000 + 0.0000j,
            -0.9239 - 0.3827j,
            -0.7071 - 0.7071j,
            -0.3827 - 0.9239j,
            -0.0000 - 1.0000j,
            0.3827 - 0.9239j,
            0.7071 - 0.7071j,
            0.9239 - 0.3827j,
        ]
    )
    np.testing.assert_almost_equal(psk.symbol_map, symbol_map, decimal=4)


def test_16psk_symbol_map_gray():
    """
    Matlab:
        >> M = 16; transpose(pskmod(0:M-1, M, 0, 'gray'))
    """
    psk = sdr.PSK(16, symbol_labels="gray")
    symbol_map = np.array(
        [
            1.0000 + 0.0000j,
            0.9239 + 0.3827j,
            0.3827 + 0.9239j,
            0.7071 + 0.7071j,
            -0.9239 + 0.3827j,
            -0.7071 + 0.7071j,
            0.0000 + 1.0000j,
            -0.3827 + 0.9239j,
            0.9239 - 0.3827j,
            0.7071 - 0.7071j,
            -0.0000 - 1.0000j,
            0.3827 - 0.9239j,
            -1.0000 + 0.0000j,
            -0.9239 - 0.3827j,
            -0.3827 - 0.9239j,
            -0.7071 - 0.7071j,
        ]
    )
    np.testing.assert_almost_equal(psk.symbol_map, symbol_map, decimal=4)
