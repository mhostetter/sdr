import numpy as np
import pytest

import sdr
import sdr._helper


def test_verify_positional_args():
    x, y = 1, "two"
    sdr._helper.verify_positional_args((x, y), 2)

    with pytest.raises(ValueError):
        sdr._helper.verify_positional_args((x, y), 1)


def test_verify_specified():
    x = 1
    sdr._helper.verify_specified(x)

    with pytest.raises(ValueError):
        x = None
        sdr._helper.verify_specified(None)


def test_verify_not_specified():
    x = None
    sdr._helper.verify_not_specified(x)

    with pytest.raises(ValueError):
        x = 1
        sdr._helper.verify_not_specified(x)


def test_verify_only_one_specified():
    x, y = 1, None
    sdr._helper.verify_only_one_specified(x, y)

    with pytest.raises(ValueError):
        x, y = 1, 2
        sdr._helper.verify_only_one_specified(x, y)

    with pytest.raises(ValueError):
        x, y = None, None
        sdr._helper.verify_only_one_specified(x, y)


def test_verify_at_least_one_specified():
    x, y = 1, None
    sdr._helper.verify_at_least_one_specified(x, y)

    x, y = 1, "two"
    sdr._helper.verify_at_least_one_specified(x, y)

    with pytest.raises(ValueError):
        x, y = None, None
        sdr._helper.verify_at_least_one_specified(x, y)


def test_verify_isinstance():
    x = 1
    sdr._helper.verify_isinstance(x, int)

    x = "one"
    sdr._helper.verify_isinstance(x, str)

    with pytest.raises(TypeError):
        x = 1
        sdr._helper.verify_isinstance(x, str)

    with pytest.raises(TypeError):
        x = "one"
        sdr._helper.verify_isinstance(x, int)

    with pytest.raises(TypeError):
        x = 3.14
        sdr._helper.verify_isinstance(x, (int, str))


def test_verify_scalar():
    sdr._helper.verify_scalar(None, optional=True, int=True)
    with pytest.raises(TypeError):
        sdr._helper.verify_scalar(None, int=True)

    sdr._helper.verify_scalar(1, int=True)
    with pytest.raises(TypeError):
        sdr._helper.verify_scalar(3.14, int=True)

    sdr._helper.verify_scalar(3.14, float=True)
    with pytest.raises(TypeError):
        sdr._helper.verify_scalar(3.14 + 1j, float=True)

    sdr._helper.verify_scalar(3.14 + 1j, complex=True)
    with pytest.raises(TypeError):
        sdr._helper.verify_scalar("3.14", complex=True)

    sdr._helper.verify_scalar(3.14, real=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_scalar(3.14 + 1j, real=True)

    sdr._helper.verify_scalar(3.14 + 1j, imaginary=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_scalar("3.14", imaginary=True)

    sdr._helper.verify_scalar(-1, negative=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_scalar(1, negative=True)

    sdr._helper.verify_scalar(0, non_negative=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_scalar(-1, non_negative=True)

    sdr._helper.verify_scalar(1, positive=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_scalar(0, positive=True)

    sdr._helper.verify_scalar(2, even=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_scalar(3, even=True)

    sdr._helper.verify_scalar(3, odd=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_scalar(2, odd=True)

    sdr._helper.verify_scalar(8, power_of_two=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_scalar(7, power_of_two=True)

    sdr._helper.verify_scalar(3, inclusive_min=3)
    with pytest.raises(ValueError):
        sdr._helper.verify_scalar(2, inclusive_min=3)

    sdr._helper.verify_scalar(3, inclusive_max=3)
    with pytest.raises(ValueError):
        sdr._helper.verify_scalar(4, inclusive_max=3)

    sdr._helper.verify_scalar(4, exclusive_min=3)
    with pytest.raises(ValueError):
        sdr._helper.verify_scalar(3, exclusive_min=3)

    sdr._helper.verify_scalar(3, exclusive_max=4)
    with pytest.raises(ValueError):
        sdr._helper.verify_scalar(4, exclusive_max=4)

    # TODO: Test accept NumPy and convert NumPy


def test_verify_arraylike():
    sdr._helper.verify_arraylike(None, optional=True, int=True)
    with pytest.raises(TypeError):
        sdr._helper.verify_arraylike(None, int=True)

    sdr._helper.verify_arraylike(1, int=True)
    with pytest.raises(TypeError):
        sdr._helper.verify_arraylike(3.14, int=True)

    sdr._helper.verify_arraylike(3.14, float=True)
    with pytest.raises(TypeError):
        sdr._helper.verify_arraylike(3.14 + 1j, float=True)

    sdr._helper.verify_arraylike(3.14 + 1j, complex=True)
    with pytest.raises(TypeError):
        sdr._helper.verify_arraylike("3.14", complex=True)

    sdr._helper.verify_arraylike(3.14, real=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_arraylike(3.14 + 1j, real=True)

    sdr._helper.verify_arraylike(3.14 + 1j, imaginary=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_arraylike("3.14", imaginary=True)

    sdr._helper.verify_arraylike(-1, negative=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_arraylike(1, negative=True)

    sdr._helper.verify_arraylike(0, non_negative=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_arraylike(-1, non_negative=True)

    sdr._helper.verify_arraylike(1, positive=True)
    with pytest.raises(ValueError):
        sdr._helper.verify_arraylike(0, positive=True)

    sdr._helper.verify_arraylike(3, inclusive_min=3)
    with pytest.raises(ValueError):
        sdr._helper.verify_arraylike(2, inclusive_min=3)

    sdr._helper.verify_arraylike(3, inclusive_max=3)
    with pytest.raises(ValueError):
        sdr._helper.verify_arraylike(4, inclusive_max=3)

    sdr._helper.verify_arraylike(4, exclusive_min=3)
    with pytest.raises(ValueError):
        sdr._helper.verify_arraylike(3, exclusive_min=3)

    sdr._helper.verify_arraylike(3, exclusive_max=4)
    with pytest.raises(ValueError):
        sdr._helper.verify_arraylike(4, exclusive_max=4)


def test_verify_bool():
    x = True
    sdr._helper.verify_bool(x)

    x = np.bool_(True)
    sdr._helper.verify_bool(x)

    with pytest.raises(TypeError):
        x = 1
        sdr._helper.verify_bool(x)

    with pytest.raises(TypeError):
        x = "one"
        sdr._helper.verify_bool(x)

    with pytest.raises(TypeError):
        x = 3.14
        sdr._helper.verify_bool(x)

    with pytest.raises(TypeError):
        x = np.bool_(True)
        sdr._helper.verify_bool(x, accept_numpy=False)


def test_verify_literal():
    x = 3.14
    sdr._helper.verify_literal(x, (3.14, 2.718))

    with pytest.raises(ValueError):
        x = 1
        sdr._helper.verify_literal(x, (2, 3))

    with pytest.raises(ValueError):
        x = "one"
        sdr._helper.verify_literal(x, ("two", "three"))


def test_verify_coprime():
    x, y = 3, 5
    sdr._helper.verify_coprime(x, y)

    with pytest.raises(ValueError):
        x, y = 3, 6
        sdr._helper.verify_coprime(x, y)


def test_verify_condition():
    x = 3
    sdr._helper.verify_condition(x > 2)

    with pytest.raises(ValueError):
        x = 1
        sdr._helper.verify_condition(x > 2)


def test_verify_same_shape():
    x = np.zeros((2, 3))
    y = np.zeros((2, 3))
    sdr._helper.verify_same_shape(x, y)

    with pytest.raises(ValueError):
        x = np.zeros((2, 3))
        y = np.zeros((2, 4))
        sdr._helper.verify_same_shape(x, y)


def test_convert_output():
    assert sdr._helper.convert_output(1) == 1
    assert sdr._helper.convert_output(np.array(1)) == 1
    assert sdr._helper.convert_output(np.array([1]), squeeze=True) == 1

    assert sdr._helper.convert_output(3.14) == 3.14
    assert sdr._helper.convert_output(np.array(3.14)) == 3.14
    assert sdr._helper.convert_output(np.array([3.14]), squeeze=True) == 3.14

    assert sdr._helper.convert_output(3.14 + 1j) == 3.14 + 1j
    assert sdr._helper.convert_output(np.array(3.14 + 1j)) == 3.14 + 1j
    assert sdr._helper.convert_output(np.array([3.14 + 1j]), squeeze=True) == 3.14 + 1j

    assert sdr._helper.convert_output(True) is True
    assert sdr._helper.convert_output(np.array(True)) is True
    assert sdr._helper.convert_output(np.array([True]), squeeze=True) is True

    x = np.array([1, 2, 3])
    assert np.array_equal(sdr._helper.convert_output(x), x)
