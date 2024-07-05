import galois
import pytest

import sdr


def test_exceptions():
    # NOTE: Need to use next() to get the exception
    with pytest.raises(TypeError):
        # Degree must be an integer
        next(sdr.preferred_pairs(6.0))
    with pytest.raises(ValueError):
        # Degree must be positive
        next(sdr.preferred_pairs(-1))

    with pytest.raises(TypeError):
        # Polynomial must be polynomial like
        next(sdr.preferred_pairs(6, poly=1.0))
    with pytest.raises(ValueError):
        # Polynomial must be a primitive polynomial
        next(sdr.preferred_pairs(6, poly=galois.Poly([1, 0, 0, 0, 0, 0, 1])))
    with pytest.raises(ValueError):
        # Polynomial must have correct degree
        next(sdr.preferred_pairs(6, poly=galois.Poly.Degrees([4, 1, 0])))


def test_degree_divides_4():
    assert list(sdr.preferred_pairs(4)) == []
    assert list(sdr.preferred_pairs(8)) == []
    assert list(sdr.preferred_pairs(12)) == []


def test_degree_3():
    assert list(sdr.preferred_pairs(3)) == [
        (galois.Poly.Str("x^3 + x + 1"), galois.Poly.Str("x^3 + x^2 + 1")),
    ]


def test_degree_5():
    assert list(sdr.preferred_pairs(5)) == [
        (galois.Poly.Str("x^5 + x^2 + 1"), galois.Poly.Str("x^5 + x^3 + x^2 + x + 1")),
        (galois.Poly.Str("x^5 + x^2 + 1"), galois.Poly.Str("x^5 + x^4 + x^2 + x + 1")),
        (galois.Poly.Str("x^5 + x^2 + 1"), galois.Poly.Str("x^5 + x^4 + x^3 + x + 1")),
        (galois.Poly.Str("x^5 + x^2 + 1"), galois.Poly.Str("x^5 + x^4 + x^3 + x^2 + 1")),
        (galois.Poly.Str("x^5 + x^3 + 1"), galois.Poly.Str("x^5 + x^3 + x^2 + x + 1")),
        (galois.Poly.Str("x^5 + x^3 + 1"), galois.Poly.Str("x^5 + x^4 + x^2 + x + 1")),
        (galois.Poly.Str("x^5 + x^3 + 1"), galois.Poly.Str("x^5 + x^4 + x^3 + x + 1")),
        (galois.Poly.Str("x^5 + x^3 + 1"), galois.Poly.Str("x^5 + x^4 + x^3 + x^2 + 1")),
        (galois.Poly.Str("x^5 + x^3 + x^2 + x + 1"), galois.Poly.Str("x^5 + x^4 + x^2 + x + 1")),
        (galois.Poly.Str("x^5 + x^3 + x^2 + x + 1"), galois.Poly.Str("x^5 + x^4 + x^3 + x + 1")),
        (galois.Poly.Str("x^5 + x^4 + x^2 + x + 1"), galois.Poly.Str("x^5 + x^4 + x^3 + x^2 + 1")),
        (galois.Poly.Str("x^5 + x^4 + x^3 + x + 1"), galois.Poly.Str("x^5 + x^4 + x^3 + x^2 + 1")),
    ]


def test_degree_5_specific():
    assert list(sdr.preferred_pairs(5, poly="x^5 + x^2 + 1")) == [
        (galois.Poly.Str("x^5 + x^2 + 1"), galois.Poly.Str("x^5 + x^3 + x^2 + x + 1")),
        (galois.Poly.Str("x^5 + x^2 + 1"), galois.Poly.Str("x^5 + x^4 + x^2 + x + 1")),
        (galois.Poly.Str("x^5 + x^2 + 1"), galois.Poly.Str("x^5 + x^4 + x^3 + x + 1")),
        (galois.Poly.Str("x^5 + x^2 + 1"), galois.Poly.Str("x^5 + x^4 + x^3 + x^2 + 1")),
    ]
