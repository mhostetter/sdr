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
        # Index must be an integer
        sdr.m_sequence(6, index=1.0)
    with pytest.raises(ValueError):
        # Index must be in [1, q^n)
        sdr.m_sequence(6, index=0)
    with pytest.raises(ValueError):
        # Index must be in [1, q^n)
        sdr.m_sequence(6, index=2**6)


def verify_code(degree, poly, code_truth):
    code_truth = np.array(code_truth)

    decimal = sdr.m_sequence(degree, poly=poly, output="decimal")
    assert isinstance(decimal, np.ndarray)
    assert np.array_equal(decimal, code_truth)

    field = sdr.m_sequence(degree, poly=poly, output="field")
    assert isinstance(field, galois.FieldArray)
    assert np.array_equal(field, code_truth)

    bipolar = sdr.m_sequence(degree, poly=poly, output="bipolar")
    assert isinstance(bipolar, np.ndarray)
    assert np.array_equal(bipolar, 1 - 2 * code_truth)


def test_degree_2():
    """
    MATLAB:
        >> pn = comm.PNSequence('Polynomial', [2 1 0], 'InitialConditions', [0 1], 'SamplesPerFrame', 2^2 - 1);
        >> pn()
    """
    poly = galois.Poly.Degrees([2, 1, 0])
    code_truth = np.array([1, 0, 1])
    verify_code(2, poly, code_truth)


def test_degree_3():
    """
    MATLAB:
        >> pn = comm.PNSequence('Polynomial', [3 1 0], 'InitialConditions', [0 0 1], 'SamplesPerFrame', 2^3 - 1);
        >> pn()
    """
    poly = galois.Poly.Degrees([3, 1, 0])
    code_truth = np.array([1, 0, 0, 1, 0, 1, 1])
    verify_code(3, poly, code_truth)


def test_degree_4():
    """
    MATLAB:
        >> pn = comm.PNSequence('Polynomial', [4 1 0], 'InitialConditions', [0 0 0 1], 'SamplesPerFrame', 2^4 - 1);
        >> pn()
    """
    poly = galois.Poly.Degrees([4, 1, 0])
    code_truth = np.array([1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    verify_code(4, poly, code_truth)


def test_degree_5():
    """
    MATLAB:
        >> pn = comm.PNSequence('Polynomial', [5 2 0], 'InitialConditions', [0 0 0 0 1], 'SamplesPerFrame', 2^5 - 1);
        >> pn()
    """
    poly = galois.Poly.Degrees([5, 2, 0])
    code_truth = np.array([1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0])
    verify_code(5, poly, code_truth)


def test_degree_6():
    """
    MATLAB:
        >> pn = comm.PNSequence('Polynomial', [6 1 0], 'InitialConditions', [0 0 0 0 0 1], 'SamplesPerFrame', 2^6 - 1);
        >> pn()
    """
    poly = galois.Poly.Degrees([6, 1, 0])
    code_truth = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1])  # fmt: skip
    verify_code(6, poly, code_truth)
