import numpy as np

import sdr


def verify_code(length, poly1, poly2, index, code_truth):
    code_truth = np.array(code_truth)

    decimal = sdr.gold_code(length, index=index, poly1=poly1, poly2=poly2, verify=False, output="binary")
    assert isinstance(decimal, np.ndarray)
    assert np.array_equal(decimal, code_truth)


def test_degree_3():
    """
    MATLAB:
        >> gold = comm.GoldSequence('FirstPolynomial', 'x^3 + x + 1', ...
            'SecondPolynomial', 'x^3 + x^2 + 1', ...
            'FirstInitialConditions', [0 0 1], ...
            'SecondInitialConditions', [0 0 1], ...
            'Index', -2, ...
            'SamplesPerFrame', 2^3 - 1);
        >> gold()
    """
    verify_code(
        7,
        "x^3 + x + 1",
        "x^3 + x^2 + 1",
        -2,
        [1, 0, 0, 1, 0, 1, 1],
    )
    verify_code(
        7,
        "x^3 + x + 1",
        "x^3 + x^2 + 1",
        -1,
        [1, 0, 0, 1, 1, 1, 0],
    )
    verify_code(
        7,
        "x^3 + x + 1",
        "x^3 + x^2 + 1",
        0,
        [0, 0, 0, 0, 1, 0, 1],
    )
    verify_code(
        7,
        "x^3 + x + 1",
        "x^3 + x^2 + 1",
        1,
        [1, 0, 1, 0, 1, 1, 0],
    )
    verify_code(
        7,
        "x^3 + x + 1",
        "x^3 + x^2 + 1",
        2,
        [1, 1, 1, 0, 0, 0, 1],
    )


def test_degree_4():
    """
    MATLAB:
        >> gold = comm.GoldSequence('FirstPolynomial', 'x^4 + x + 1', ...
            'SecondPolynomial', 'x^4 + x^3 + 1', ...
            'FirstInitialConditions', [0 0 0 1], ...
            'SecondInitialConditions', [0 0 0 1], ...
            'Index', -2, ...
            'SamplesPerFrame', 2^4 - 1);
        >> gold()
    """
    # These actually aren't really Gold codes. I needed to disable the verification.
    verify_code(
        15,
        "x^4 + x + 1",
        "x^4 + x^3 + 1",
        -2,
        [1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
    )
    verify_code(
        15,
        "x^4 + x + 1",
        "x^4 + x^3 + 1",
        -1,
        [1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    )
    verify_code(
        15,
        "x^4 + x + 1",
        "x^4 + x^3 + 1",
        0,
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    )
    verify_code(
        15,
        "x^4 + x + 1",
        "x^4 + x^3 + 1",
        1,
        [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    )
    verify_code(
        15,
        "x^4 + x + 1",
        "x^4 + x^3 + 1",
        2,
        [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
    )


def test_degree_5():
    """
    MATLAB:
        >> gold = comm.GoldSequence('FirstPolynomial', 'x^5 + x^2 + 1', ...
            'SecondPolynomial', 'x^5 + x^3 + 1', ...
            'FirstInitialConditions', [0 0 0 0 1], ...
            'SecondInitialConditions', [0 0 0 0 1], ...
            'Index', -2, ...
            'SamplesPerFrame', 2^5 - 1);
        >> gold()
    """
    # These actually aren't really Gold codes. I needed to disable the verification.
    verify_code(
        31,
        "x^5 + x^2 + 1",
        "x^5 + x^3 + 1",
        -2,
        [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    )
    verify_code(
        31,
        "x^5 + x^2 + 1",
        "x^5 + x^3 + 1",
        -1,
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0],
    )
    verify_code(
        31,
        "x^5 + x^2 + 1",
        "x^5 + x^3 + 1",
        0,
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    )
    verify_code(
        31,
        "x^5 + x^2 + 1",
        "x^5 + x^3 + 1",
        1,
        [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
    )
    verify_code(
        31,
        "x^5 + x^2 + 1",
        "x^5 + x^3 + 1",
        2,
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    )
