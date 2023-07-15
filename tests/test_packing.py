import numpy as np
import pytest

import sdr


def test_pack_exceptions():
    with pytest.raises(ValueError):
        x = np.array([1, 0, 0, 0, 1, 1, 0, 1], dtype=np.float32)
        sdr.pack(x, 2)
    with pytest.raises(ValueError):
        x = np.array([1, 0, 0, 0, 1, 1, 0, 1], dtype=int)
        sdr.pack(x, 0)


def test_pack_1d():
    # 32 elements
    x = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
    assert np.array_equal(sdr.pack(x, 1), x)
    assert np.array_equal(sdr.pack(x, 2), [2, 0, 3, 1, 3, 0, 2, 3, 0, 2, 3, 1, 1, 3, 1, 0])
    assert np.array_equal(sdr.pack(x, 3), [4, 3, 3, 4, 5, 4, 5, 5, 3, 5, 0])
    assert np.array_equal(sdr.pack(x, 4), [8, 13, 12, 11, 2, 13, 7, 4])
    assert np.array_equal(sdr.pack(x, 5), [17, 23, 5, 18, 26, 29, 0])
    assert np.array_equal(sdr.pack(x, 6), [35, 28, 44, 45, 29, 0])
    assert np.array_equal(sdr.pack(x, 7), [70, 114, 101, 87, 32])
    assert np.array_equal(sdr.pack(x, 8), [141, 203, 45, 116])
    assert np.array_equal(sdr.pack(x, 11), [1134, 715, 744])
    assert np.array_equal(sdr.pack(x, 16), [36299, 11636])


def test_pack_2d():
    # 2 x 16 elements
    x = [[1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]]
    assert np.array_equal(sdr.pack(x, 1), x)
    assert np.array_equal(sdr.pack(x, 2), [[3, 1, 0, 3, 2, 1, 0, 3], [2, 2, 3, 3, 3, 3, 1, 2]])
    assert np.array_equal(sdr.pack(x, 3), [[6, 4, 7, 1, 1, 4], [5, 3, 7, 7, 3, 0]])
    assert np.array_equal(sdr.pack(x, 4), [[13, 3, 9, 3], [10, 15, 15, 6]])
    assert np.array_equal(sdr.pack(x, 5), [[26, 14, 9, 16], [21, 31, 27, 0]])
    assert np.array_equal(sdr.pack(x, 6), [[52, 57, 12], [43, 63, 24]])
    assert np.array_equal(sdr.pack(x, 7), [[105, 100, 96], [87, 125, 64]])
    assert np.array_equal(sdr.pack(x, 8), [[211, 147], [175, 246]])
    assert np.array_equal(sdr.pack(x, 11), [[1692, 1216], [1407, 1408]])
    assert np.array_equal(sdr.pack(x, 16), [[54163], [45046]])


def test_pack_dtype():
    x = np.random.randint(0, 2, 50)
    assert sdr.pack(x, 7).dtype == np.uint8
    assert sdr.pack(x, 8).dtype == np.uint8
    assert sdr.pack(x, 9).dtype == np.uint16
    assert sdr.pack(x, 17).dtype == np.uint32

    assert sdr.pack(x, 7, dtype=np.uint16).dtype == np.uint16


def test_unpack_exceptions():
    with pytest.raises(ValueError):
        x = np.array([8, 13, 12], dtype=np.float32)
        sdr.pack(x, 4)
    with pytest.raises(ValueError):
        x = np.array([8, 13, 12], dtype=int)
        sdr.pack(x, 0)


def test_unpack_1d():
    # 32 elements
    x = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
    assert np.array_equal(sdr.unpack(x, 1), x)
    assert np.array_equal(sdr.unpack([2, 0, 3, 1, 3, 0, 2, 3, 0, 2, 3, 1, 1, 3, 1, 0], 2), x)
    assert np.array_equal(sdr.unpack([4, 3, 3, 4, 5, 4, 5, 5, 3, 5, 0], 3), x + [0])
    assert np.array_equal(sdr.unpack([8, 13, 12, 11, 2, 13, 7, 4], 4), x)
    assert np.array_equal(sdr.unpack([17, 23, 5, 18, 26, 29, 0], 5), x + [0] * 3)
    assert np.array_equal(sdr.unpack([35, 28, 44, 45, 29, 0], 6), x + [0] * 4)
    assert np.array_equal(sdr.unpack([70, 114, 101, 87, 32], 7), x + [0] * 3)
    assert np.array_equal(sdr.unpack([141, 203, 45, 116], 8), x)
    assert np.array_equal(sdr.unpack([1134, 715, 744], 11), x + [0])
    assert np.array_equal(sdr.unpack([36299, 11636], 16), x)


def test_unpack_2d():
    # 2 x 16 elements
    x1 = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1]
    x2 = [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]
    x = [x1, x2]
    assert np.array_equal(sdr.unpack(x, 1), x)
    assert np.array_equal(sdr.unpack([[3, 1, 0, 3, 2, 1, 0, 3], [2, 2, 3, 3, 3, 3, 1, 2]], 2), x)
    assert np.array_equal(sdr.unpack([[6, 4, 7, 1, 1, 4], [5, 3, 7, 7, 3, 0]], 3), [x1 + [0] * 2, x2 + [0] * 2])
    assert np.array_equal(sdr.unpack([[13, 3, 9, 3], [10, 15, 15, 6]], 4), x)
    assert np.array_equal(sdr.unpack([[26, 14, 9, 16], [21, 31, 27, 0]], 5), [x1 + [0] * 4, x2 + [0] * 4])
    assert np.array_equal(sdr.unpack([[52, 57, 12], [43, 63, 24]], 6), [x1 + [0] * 2, x2 + [0] * 2])
    assert np.array_equal(sdr.unpack([[105, 100, 96], [87, 125, 64]], 7), [x1 + [0] * 5, x2 + [0] * 5])
    assert np.array_equal(sdr.unpack([[211, 147], [175, 246]], 8), x)
    assert np.array_equal(sdr.unpack([[1692, 1216], [1407, 1408]], 11), [x1 + [0] * 6, x2 + [0] * 6])
    assert np.array_equal(sdr.unpack([[54163], [45046]], 16), x)


def test_unpack_dtype():
    x = np.random.randint(0, 2**7, 5, dtype=int)
    assert sdr.unpack(x, 7).dtype == np.uint8
    x = np.random.randint(0, 2**17, 5, dtype=int)
    assert sdr.unpack(x, 17).dtype == np.uint8

    assert sdr.unpack(x, 7, dtype=np.uint16).dtype == np.uint16


def test_hexdump_exceptions():
    with pytest.raises(ValueError):
        # Must only have integer arrays
        sdr.hexdump(np.array([8, 13, 12], dtype=np.float32))
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


def test_hexdump():
    hexdump = sdr.hexdump(b"The quick brown fox jumps over the lazy dog")
    hexdump_truth = "00000000  54 68 65 20 71 75 69 63 6b 20 62 72 6f 77 6e 20  The quick brown \n00000010  66 6f 78 20 6a 75 6d 70 73 20 6f 76 65 72 20 74  fox jumps over t\n00000020  68 65 20 6c 61 7a 79 20 64 6f 67                 he lazy dog\n"
    assert hexdump == hexdump_truth

    hexdump = sdr.hexdump([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], width=4)
    hexdump_truth = "00000000  01 02 03 04  ....\n00000004  05 06 07 08  ....\n00000008  09 0a        ..\n"
    assert hexdump == hexdump_truth
