import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # Must only have integer arrays
        sdr.hexdump(np.array([8, 13, 12], dtype=float))
    with pytest.raises(ValueError):
        # Must only have 1D arrays
        sdr.hexdump(np.array([[8, 13, 12]], dtype=int))
    with pytest.raises(ValueError):
        # Must only have values 0-255
        sdr.hexdump([8, 256, 12])
    with pytest.raises(ValueError):
        # Must only have values 0-255
        sdr.hexdump([8, -1, 12])
    with pytest.raises(ValueError):
        # Must have width in 1-16
        sdr.hexdump([8, 13, 12], width=0)
    with pytest.raises(ValueError):
        # Must have width in 1-16
        sdr.hexdump([8, 13, 12], width=17)


def test_output():
    hexdump = sdr.hexdump(b"The quick brown fox jumps over the lazy dog")
    hexdump_truth = "00000000  54 68 65 20 71 75 69 63 6b 20 62 72 6f 77 6e 20  The quick brown \n00000010  66 6f 78 20 6a 75 6d 70 73 20 6f 76 65 72 20 74  fox jumps over t\n00000020  68 65 20 6c 61 7a 79 20 64 6f 67                 he lazy dog\n"
    assert hexdump == hexdump_truth

    hexdump = sdr.hexdump([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], width=4)
    hexdump_truth = "00000000  01 02 03 04  ....\n00000004  05 06 07 08  ....\n00000008  09 0a        ..\n"
    assert hexdump == hexdump_truth
