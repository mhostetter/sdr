import galois

import sdr


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
