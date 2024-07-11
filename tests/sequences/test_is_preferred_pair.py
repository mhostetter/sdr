import sdr


def test_degree_divides_4():
    assert sdr.is_preferred_pair("x^4 + x + 1", "x^4 + x^3 + 1") is False


def test_degree_3():
    assert sdr.is_preferred_pair("x^3 + x + 1", "x^3 + x^2 + 1") is True


def test_degree_5():
    assert sdr.is_preferred_pair("x^5 + x^2 + 1", "x^5 + x^3 + x^2 + x + 1") is True
    assert sdr.is_preferred_pair("x^5 + x^2 + 1", "x^5 + x^3 + 1") is False
