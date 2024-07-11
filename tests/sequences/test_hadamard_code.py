"""
MATLAB:
    for N = 1:4
        for idx = 0:2^N-1
            walsh = comm.HadamardCode("Length", 2^N, "SamplesPerFrame", 2^N, "Index", idx);
            disp(N)
            disp(idx)
            disp(walsh())
        end
    end
"""

import numpy as np

import sdr


def verify_code(length, index, sequence_truth):
    sequence_truth = np.array(sequence_truth)

    sequence = sdr.hadamard_code(length, index, output="bipolar")
    assert isinstance(sequence, np.ndarray)
    assert np.array_equal(sequence, sequence_truth)

    code = sdr.hadamard_code(length, index, output="binary")
    assert isinstance(code, np.ndarray)
    assert np.array_equal(code, (1 - sequence_truth) // 2)


def test_length_2():
    verify_code(2, 0, [1, 1])
    verify_code(2, 1, [1, -1])


def test_length_4():
    verify_code(4, 0, [1, 1, 1, 1])
    verify_code(4, 1, [1, -1, 1, -1])
    verify_code(4, 2, [1, 1, -1, -1])
    verify_code(4, 3, [1, -1, -1, 1])


def test_length_8():
    verify_code(8, 0, [1, 1, 1, 1, 1, 1, 1, 1])
    verify_code(8, 1, [1, -1, 1, -1, 1, -1, 1, -1])
    verify_code(8, 2, [1, 1, -1, -1, 1, 1, -1, -1])
    verify_code(8, 3, [1, -1, -1, 1, 1, -1, -1, 1])
    verify_code(8, 4, [1, 1, 1, 1, -1, -1, -1, -1])
    verify_code(8, 5, [1, -1, 1, -1, -1, 1, -1, 1])
    verify_code(8, 6, [1, 1, -1, -1, -1, -1, 1, 1])
    verify_code(8, 7, [1, -1, -1, 1, -1, 1, 1, -1])


def test_length_16():
    verify_code(16, 0, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    verify_code(16, 1, [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    verify_code(16, 2, [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
    verify_code(16, 3, [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])
    verify_code(16, 4, [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1])
    verify_code(16, 5, [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1])
    verify_code(16, 6, [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1])
    verify_code(16, 7, [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1])
    verify_code(16, 8, [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1])
    verify_code(16, 9, [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1])
    verify_code(16, 10, [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1])
    verify_code(16, 11, [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1])
    verify_code(16, 12, [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1])
    verify_code(16, 13, [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1])
    verify_code(16, 14, [1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1])
    verify_code(16, 15, [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1])
