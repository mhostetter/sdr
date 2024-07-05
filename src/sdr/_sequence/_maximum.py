"""
A module containing maximum-length sequences.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterator, overload

import galois
import numpy as np
import numpy.typing as npt
from galois import Poly
from galois.typing import PolyLike
from typing_extensions import Literal

from .._helper import export
from ._conversion import code_to_sequence, field_to_code
from ._lfsr import FLFSR


@overload
def m_sequence(
    degree: int,
    poly: PolyLike | None = None,
    index: int = 1,
    output: Literal["decimal"] = "decimal",
) -> npt.NDArray[np.int_]: ...


@overload
def m_sequence(
    degree: int,
    poly: PolyLike | None = None,
    index: int = 1,
    output: Literal["field"] = "decimal",
) -> galois.FieldArray: ...


@overload
def m_sequence(
    degree: int,
    poly: PolyLike | None = None,
    index: int = 1,
    output: Literal["bipolar"] = "decimal",
) -> npt.NDArray[np.float64]: ...


@export
def m_sequence(
    degree: Any,
    poly: Any = None,
    index: Any = 1,
    output: Any = "decimal",
) -> Any:
    r"""
    Generates a maximum-length sequence ($m$-sequence) from a Fibonacci linear feedback shift register (LFSR).

    Arguments:
        degree: The degree $m$ of the LFSR.
        poly: The characteristic polynomial of the LFSR over $\mathrm{GF}(q)$. The default is `None`, which uses the
            primitive polynomial of degree $m$ over $\mathrm{GF}(2)$, `galois.primitive_poly(2, degree)`.
        index: The index $i$ in $[1, q^m)$ of the $m$-sequence. The index represents the initial state of the LFSR.
            The index dictates the phase of the $m$-sequence. The integer index is interpreted as a polynomial over
            $\mathrm{GF}(q)$, whose coefficients are the shift register values. The default is 1, which corresponds
            to the $[0, \dots, 0, 1]$ state.
        output: The output format of the $m$-sequence.

            - `"decimal"` (default): The $m$-sequence with decimal values in $[0, q^m)$.
            - `"field"`: The $m$-sequence as a Galois field array over $\mathrm{GF}(q^m)$.
            - `"bipolar"`: The $m$-sequence with bipolar values of 1 and -1. Only valid for $q = 2$.

    Returns:
        The length-$q^m - 1$ $m$-sequence.

    References:
        - https://en.wikipedia.org/wiki/Maximum_length_sequence

    Examples:
        Generate a maximum-length sequence of degree-4 over $\mathrm{GF}(2)$.

        .. ipython:: python

            sdr.m_sequence(4)
            sdr.m_sequence(4, output="bipolar")
            sdr.m_sequence(4, output="field")

        Compare the sequence with index 1 to the sequence with index 2. They are just phase shifts of each other.

        .. ipython:: python

            sdr.m_sequence(4, index=2)

        Generate a maximum-length sequence of degree-4 over $\mathrm{GF}(3^2)$.

        .. ipython:: python

            c = galois.primitive_poly(3**2, 4); c
            x = sdr.m_sequence(4, poly=c); x
            x.size

        Plot the auto-correlation of a length-63 $m$-sequence. Notice that the linear correlation produces sidelobes
        for non-zero lag. However, the circular correlation only produces magnitudes of 1 for non-zero lag.

        .. ipython:: python

            x = sdr.m_sequence(6, output="bipolar")

            @savefig sdr_m_sequence_1.png
            plt.figure(); \
            sdr.plot.correlation(x, x, mode="circular"); \
            plt.ylim(0, 63);

        The cross-correlation of two $m$-sequences with different indices is low for zero lag. However, for non-zero
        lag the cross-correlation is very large.

        .. ipython:: python

            x = sdr.m_sequence(6, index=1, output="bipolar")
            y = sdr.m_sequence(6, index=30, output="bipolar")

            @savefig sdr_m_sequence_2.png
            plt.figure(); \
            sdr.plot.correlation(x, y, mode="circular"); \
            plt.ylim(0, 63);

    Group:
        sequences-maximum-length
    """
    if not isinstance(degree, int):
        raise TypeError(f"Argument 'degree' must be an integer, not {type(degree)}.")
    if not degree > 0:
        raise ValueError(f"Argument 'degree' must be positive, not {degree}.")

    if poly is None:
        poly = galois.primitive_poly(2, degree)  # Characteristic polynomial
    elif isinstance(poly, Poly):
        pass
    else:
        poly = Poly._PolyLike(poly, field=galois.GF(2))
    if not poly.degree == degree:
        raise ValueError(f"Argument 'poly' must be a polynomial of degree {degree}, not {poly.degree}.")
    if not poly.is_primitive():
        raise ValueError(f"Argument 'poly' must be a primitive polynomial, {poly} is not.")
    q = poly.field.order

    if not isinstance(index, int):
        raise TypeError(f"Argument 'index' must be an integer, not {type(index)}.")
    if not 1 <= index < q**degree:
        raise ValueError(f"Argument 'index' must be in [1, q^m), not {index}.")
    state_poly = Poly.Int(index, field=poly.field)
    state_vector = state_poly.coefficients(degree)

    if output not in ["decimal", "field", "bipolar"]:
        raise ValueError(f"Argument 'output' must be 'decimal', 'field', or 'bipolar', not {output}.")

    lfsr = FLFSR(poly, state=state_vector)
    code = lfsr.step(q**degree - 1)

    if output == "decimal":
        return field_to_code(code)
    elif output == "field":
        return code
    else:
        if not q == 2:
            raise ValueError(f"Argument 'output' can only be 'bipolar' when q = 2, not {q}.")
        return code_to_sequence(field_to_code(code))


@export
def preferred_pairs(
    degree: int,
    poly: PolyLike | None = None,
) -> Iterator[tuple[Poly, Poly]]:
    r"""
    Generates primitive polynomials of degree $m$ that produce preferred pair $m$-sequences.

    Arguments:
        degree: The degree $m$ of the $m$-sequences.
        poly: The first polynomial $f(x)$ in the preferred pair. If `None`, all primitive polynomials of degree $m$
            that yield preferred pair $m$-sequences are returned.

    Returns:
        An iterator of primitive polynomials $(f(x), g(x))$ of degree $m$ that produce preferred pair $m$-sequences.

    See Also:
        sdr.is_preferred_pair, sdr.gold_code

    Notes:
        A preferred pair of primitive polynomials of degree $m$ are two polynomials $f(x)$ and $g(x)$ such that the
        periodic cross-correlation of the two $m$-sequences generated by $f(x)$ and $g(x)$ have only the values in
        $\{-1, -t(m), t(m) - 2\}$.

        $$
        t(m) = \begin{cases}
        2^{(m+1)/2} + 1 & \text{if $m$ is odd} \\
        2^{(m+2)/2} + 1 & \text{if $m$ is even}
        \end{cases}
        $$

        There are no preferred pairs for degrees divisible by 4.

    References:
        - John Proakis, *Digital Communications*, Chapter 12.2-5: Generation of PN Sequences.

    Examples:
        Generate one preferred pair with $f(x) = x^5 + x^3 + 1$.

        .. ipython:: python

            next(sdr.preferred_pairs(5, poly="x^5 + x^3 + 1"))

        Generate all preferred pairs with $f(x) = x^5 + x^3 + 1$.

        .. ipython:: python

            list(sdr.preferred_pairs(5, poly="x^5 + x^3 + 1"))

        Generate all preferred pairs with degree 5.

        .. ipython:: python

            list(sdr.preferred_pairs(5))

        Note, there are no preferred pairs for degrees divisible by 4.

        .. ipython:: python

            list(sdr.preferred_pairs(4))
            list(sdr.preferred_pairs(8))

    Group:
        sequences-maximum-length
    """
    if not isinstance(degree, int):
        raise TypeError(f"Argument 'degree' must be an integer, not {type(degree)}.")
    if not degree > 0:
        raise ValueError(f"Argument 'degree' must be positive, not {degree}.")

    if degree % 4 == 0:
        # There are no preferred pairs for degrees divisible by 4
        return

    # Compute t(m) for degree m, Equation 12.2-73
    if degree % 2 == 1:
        t_m = 2 ** ((degree + 1) // 2) + 1
    else:
        t_m = 2 ** ((degree + 2) // 2) + 1

    # Determine the valid cross-correlation values for preferred pairs, Page 799
    valid_values = [-1, -t_m, t_m - 2]

    if poly is None:
        # Find all combinations of primitive polynomials of degree m
        for poly1, poly2 in itertools.combinations(galois.primitive_polys(2, degree), 2):
            # Create first m-sequence with the first polynomial
            u = m_sequence(degree, poly1, output="bipolar")
            U = np.fft.fft(u)

            # Create second m-sequence with the second polynomial
            v = m_sequence(degree, poly2, output="bipolar")
            V = np.fft.fft(v)

            # Compute the periodic cross-correlation of the two m-sequences
            pccf = np.fft.ifft(U * V.conj()).real  # The inputs are real, so the output is real
            pccf = np.around(pccf).astype(int)  # Round to the nearest integer so isin() works

            if np.all(np.isin(pccf, valid_values)):
                yield poly1, poly2
    else:
        # Find all combinations of the first polynomial with all primitive polynomials of degree m
        poly1 = Poly._PolyLike(poly, field=galois.GF(2))
        if not poly1.degree == degree:
            raise ValueError(f"Argument 'poly1' must be a polynomial of degree {degree}, not {poly1.degree}.")
        if not poly1.is_primitive():
            raise ValueError(f"Argument 'poly1' must be a primitive polynomial, {poly1} is not.")

        # Create first m-sequence with the first polynomial
        u = m_sequence(degree, poly1, output="bipolar")
        U = np.fft.fft(u)

        for poly2 in galois.primitive_polys(2, degree):
            # Create second m-sequence with the second polynomial
            v = m_sequence(degree, poly2, output="bipolar")
            V = np.fft.fft(v)

            # Compute the periodic cross-correlation of the two m-sequences
            pccf = np.fft.ifft(U * V.conj()).real  # The inputs are real, so the output is real
            pccf = np.around(pccf).astype(int)  # Round to the nearest integer so isin() works

            if np.all(np.isin(pccf, valid_values)):
                yield poly1, poly2


@export
def is_preferred_pair(
    poly1: PolyLike,
    poly2: PolyLike,
) -> bool:
    r"""
    Determines if two primitive polynomials generate preferred pair $m$-sequences.

    Arguments:
        poly1: The first primitive polynomial $f(x)$.
        poly2: The second primitive polynomial $g(x)$.

    Returns:
        A boolean indicating if the two primitive polynomials generate preferred pair $m$-sequences.

    See Also:
        sdr.preferred_pairs, sdr.gold_code

    Notes:
        A preferred pair of primitive polynomials of degree $m$ are two polynomials $f(x)$ and $g(x)$ such that the
        periodic cross-correlation of the two $m$-sequences generated by $f(x)$ and $g(x)$ have only the values in
        $\{-1, -t(m), t(m) - 2\}$.

        $$
        t(m) = \begin{cases}
        2^{(m+1)/2} + 1 & \text{if $m$ is odd} \\
        2^{(m+2)/2} + 1 & \text{if $m$ is even}
        \end{cases}
        $$

        There are no preferred pairs for degrees divisible by 4.

    References:
        - John Proakis, *Digital Communications*, Chapter 12.2-5: Generation of PN Sequences.

    Examples:
        Determine if the pair $f(x) = x^5 + x^3 + 1$ and $g(x) = x^5 + x^3 + x^2 + x + 1$ is a preferred pair.

        .. ipython:: python

            sdr.is_preferred_pair("x^5 + x^3 + 1", "x^5 + x^3 + x^2 + x + 1")

        Determine if the pair $f(x) = x^5 + x^3 + 1$ and $g(x) = x^5 + x^2 + 1$ is a preferred pair.

        .. ipython:: python

            sdr.is_preferred_pair("x^5 + x^3 + 1", "x^5 + x^2 + 1")

    Group:
        sequences-maximum-length
    """
    poly1 = Poly._PolyLike(poly1, field=galois.GF(2))
    poly2 = Poly._PolyLike(poly2, field=galois.GF(2))
    if not poly1.degree == poly2.degree:
        raise ValueError(
            f"Arguments 'poly1' and 'poly2' must have the same degree, not {poly1.degree} and {poly2.degree}."
        )
    if not poly1.is_primitive():
        raise ValueError(f"Argument 'poly1' must be a primitive polynomial, {poly1} is not.")
    if not poly2.is_primitive():
        raise ValueError(f"Argument 'poly2' must be a primitive polynomial, {poly2} is not.")

    degree = poly1.degree
    if degree % 4 == 0:
        # There are no preferred pairs for degrees divisible by 4
        return False

    # Compute t(m) for degree m, Equation 12.2-73
    if degree % 2 == 1:
        t_m = 2 ** ((degree + 1) // 2) + 1
    else:
        t_m = 2 ** ((degree + 2) // 2) + 1

    # Determine the valid cross-correlation values for preferred pairs, Page 799
    valid_values = [-1, -t_m, t_m - 2]

    # Create first m-sequence with the first polynomial
    u = m_sequence(degree, poly1, output="bipolar")
    U = np.fft.fft(u)

    # Create second m-sequence with the second polynomial
    v = m_sequence(degree, poly2, output="bipolar")
    V = np.fft.fft(v)

    # Compute the periodic cross-correlation of the two m-sequences
    pccf = np.fft.ifft(U * V.conj()).real  # The inputs are real, so the output is real
    pccf = np.around(pccf).astype(int)  # Round to the nearest integer so isin() works

    valid = bool(np.all(np.isin(pccf, valid_values)))

    return valid
