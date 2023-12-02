import numpy as np

import sdr


def test_code_to_sequence():
    code = np.array([0, 1, 1, 0])
    sequence = sdr._sequence._code_to_sequence(code)
    sequence_truth = np.array([1, -1, -1, 1])
    assert np.array_equal(sequence, sequence_truth)


def test_sequence_to_code():
    sequence = np.array([1, -1, -1, 1])
    code = sdr._sequence._sequence_to_code(sequence)
    code_truth = np.array([0, 1, 1, 0])
    assert np.array_equal(code, code_truth)
