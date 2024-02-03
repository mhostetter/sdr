import galois
import numpy as np

import sdr


def test_ieee802_11():
    c = galois.Poly.Degrees([7, 3, 0])
    scrambler = sdr.AdditiveScrambler(c)
    x = np.zeros(127, dtype=int)  # If input is all zeros, output is the scrambling sequence
    y = scrambler.scramble(x)
    seq_truth = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # fmt: skip
    assert np.array_equal(y, seq_truth)


def test_scramble_descramble():
    c = galois.Poly.Degrees([7, 3, 0])
    scrambler = sdr.AdditiveScrambler(c)
    rng = np.random.default_rng()
    x = rng.integers(0, 2, 1000)
    y = scrambler.scramble(x)
    x_recovered = scrambler.descramble(y)
    assert np.array_equal(x, x_recovered)
