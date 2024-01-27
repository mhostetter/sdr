import galois
import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(TypeError):
        # Degree must be an integer
        sdr.m_sequence(6.0)
    with pytest.raises(ValueError):
        # Degree must be positive
        sdr.m_sequence(-1)

    with pytest.raises(TypeError):
        # Polynomial must be polynomial like
        sdr.m_sequence(6, poly=1.0)
    with pytest.raises(ValueError):
        # Polynomial must be a primitive polynomial
        sdr.m_sequence(6, poly=galois.Poly([1, 0, 0, 0, 0, 0, 1]))
    with pytest.raises(ValueError):
        # Polynomial must have correct degree
        sdr.m_sequence(6, poly=galois.Poly.Degrees([4, 1, 0]))

    with pytest.raises(TypeError):
        # State must be an integer
        sdr.m_sequence(6, state=1.0)
    with pytest.raises(ValueError):
        # State must be in [1, q^n - 1]
        sdr.m_sequence(6, state=0)
    with pytest.raises(ValueError):
        # State must be in [1, q^n - 1]
        sdr.m_sequence(6, state=2**6)


def test_degree_2():
    """
    MATLAB:
        >> pn = comm.PNSequence('Polynomial', [2 1 0], 'InitialConditions', [0 1], 'SamplesPerFrame', 2^2 - 1);
        >> pn()
    """
    # NOTE: MATLAB defines their polynomials as the characteristic, not feedback, polynomial
    poly = galois.Poly.Degrees([2, 1, 0]).reverse()
    seq_truth = np.array([1, 0, 1])
    seq = sdr.m_sequence(2, poly=poly)
    assert np.array_equal(seq, seq_truth)


def test_degree_3():
    """
    MATLAB:
        >> pn = comm.PNSequence('Polynomial', [3 1 0], 'InitialConditions', [0 0 1], 'SamplesPerFrame', 2^3 - 1);
        >> pn()
    """
    # NOTE: MATLAB defines their polynomials as the characteristic, not feedback, polynomial
    poly = galois.Poly.Degrees([3, 1, 0]).reverse()
    seq_truth = np.array([1, 0, 0, 1, 0, 1, 1])
    seq = sdr.m_sequence(3, poly=poly)
    assert np.array_equal(seq, seq_truth)


def test_degree_4():
    """
    MATLAB:
        >> pn = comm.PNSequence('Polynomial', [4 1 0], 'InitialConditions', [0 0 0 1], 'SamplesPerFrame', 2^4 - 1);
        >> pn()
    """
    # NOTE: MATLAB defines their polynomials as the characteristic, not feedback, polynomial
    poly = galois.Poly.Degrees([4, 1, 0]).reverse()
    seq_truth = np.array([1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    seq = sdr.m_sequence(4, poly=poly)
    assert np.array_equal(seq, seq_truth)


def test_degree_5():
    """
    MATLAB:
        >> pn = comm.PNSequence('Polynomial', [5 2 0], 'InitialConditions', [0 0 0 0 1], 'SamplesPerFrame', 2^5 - 1);
        >> pn()
    """
    # NOTE: MATLAB defines their polynomials as the characteristic, not feedback, polynomial
    poly = galois.Poly.Degrees([5, 2, 0]).reverse()
    seq_truth = np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0])
    seq = sdr.m_sequence(5, poly=poly)
    assert np.array_equal(seq, seq_truth)


def test_degree_6():
    """
    MATLAB:
        >> pn = comm.PNSequence('Polynomial', [6 1 0], 'InitialConditions', [0 0 0 0 0 1], 'SamplesPerFrame', 2^6 - 1);
        >> pn()
    """
    # NOTE: MATLAB defines their polynomials as the characteristic, not feedback, polynomial
    poly = galois.Poly.Degrees([6, 1, 0]).reverse()
    seq_truth = np.array(
        [
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
        ]
    )
    seq = sdr.m_sequence(6, poly=poly)
    assert np.array_equal(seq, seq_truth)
