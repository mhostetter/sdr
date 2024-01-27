"""
A module containing maximal-length sequences.
"""
from __future__ import annotations

from typing import Any, overload

import galois
import numpy as np
import numpy.typing as npt
from galois import Poly
from galois.typing import PolyLike
from typing_extensions import Literal

from .._helper import export
from ._lfsr import FLFSR


@overload
def m_sequence(
    degree: int,
    poly: PolyLike | None = None,
    state: int = 1,
    output: Literal["decimal"] = "decimal",
) -> npt.NDArray[np.int_]:
    ...


@overload
def m_sequence(
    degree: int,
    poly: PolyLike | None = None,
    state: int = 1,
    output: Literal["field"] = "decimal",
) -> galois.FieldArray:
    ...


@export
def m_sequence(
    degree: Any,
    poly: Any = None,
    state: Any = 1,
    output: Any = "decimal",
) -> Any:
    r"""
    Generates a maximal-length sequence (m-sequence) from a Fibonacci linear feedback shift register (LFSR).

    Arguments:
        degree: The degree $n$ of the LFSR.
        poly: The feedback polynomial of the LFSR over $\mathrm{GF}(q)$. Note, the feedback polynomial
            is the reciprocal of the characteristic polynomial that defines the linear recurrence relation.
            The default is `None` which uses the reciprocal primitive polynomial of degree $n$ over $\mathrm{GF}(2)$,
            `galois.primitive_poly(2, degree).reverse()`.
        state: An integer in $[1, q^{n} - 1]$ representing the initial state of the LFSR. The state corresponds to
            the phase of the m-sequence. The integer is interpreted as a polynomial over $\mathrm{GF}(q)$, whose
            coefficients are the shift register states. The default is 1, which corresponds to the
            $[0, \dots, 0, 1]$ state.
        output: The output format of the m-sequence.

            - `"decimal"` (default): The m-sequence with decimal values in $[0, q^n)$.
            - `"field"`: The m-sequence as a Galois field array over $\mathrm{GF}(q^n)$.

    Returns:
        The length-$q^n - 1$ m-sequence.

    References:
        - https://en.wikipedia.org/wiki/Maximum_length_sequence

    Examples:
        Generate a maximal-length sequence of degree-4 over $\mathrm{GF}(2)$. Compare the sequence with state 1 to
        the sequence with state 2. They are just phase shifts of each other.

        .. ipython:: python

            sdr.m_sequence(4)
            sdr.m_sequence(4, state=2)

        Generate a maximal-length sequence of degree-4 over $\mathrm{GF}(3^2)$.

        .. ipython:: python

            c = galois.primitive_poly(3**2, 4); c
            x = sdr.m_sequence(4, poly=c.reverse()); x
            x.size

    Group:
        sequences-maximal-length
    """
    if not isinstance(degree, int):
        raise TypeError(f"Argument 'degree' must be an integer, not {type(degree)}.")
    if not degree > 0:
        raise ValueError(f"Argument 'degree' must be positive, not {degree}.")

    if poly is None:
        c = galois.primitive_poly(2, degree)  # Characteristic polynomial
        poly = c.reverse()  # Feedback polynomial
    elif isinstance(poly, Poly):
        pass
    else:
        poly = Poly._PolyLike(poly, field=galois.GF(2))
    if not poly.degree == degree:
        raise ValueError(f"Argument 'poly' must be a polynomial of degree {degree}, not {poly.degree}.")
    if not poly.is_primitive():
        raise ValueError(f"Argument 'poly' must be a primitive polynomial, {poly} is not.")
    q = poly.field.order

    if not isinstance(state, int):
        raise TypeError(f"Argument 'state' must be an integer, not {type(state)}.")
    if not 1 <= state < q**degree:
        raise ValueError(f"Argument 'state' must be in [1, q^n - 1], not {state}.")
    state_poly = Poly.Int(state, field=poly.field)
    state_vector = state_poly.coefficients(degree)

    if output not in ["decimal", "field"]:
        raise ValueError(f"Argument 'output' must be 'decimal' or 'field', not {output}.")

    lfsr = FLFSR(poly, state=state_vector)
    code = lfsr.step(q**degree - 1)

    if output == "decimal":
        return code.view(np.ndarray)
    else:
        return code
