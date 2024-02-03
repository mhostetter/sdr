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
from ._conversion import code_to_sequence, field_to_code
from ._lfsr import FLFSR


@overload
def m_sequence(
    degree: int,
    poly: PolyLike | None = None,
    index: int = 1,
    output: Literal["decimal"] = "decimal",
) -> npt.NDArray[np.int_]:
    ...


@overload
def m_sequence(
    degree: int,
    poly: PolyLike | None = None,
    index: int = 1,
    output: Literal["field"] = "decimal",
) -> galois.FieldArray:
    ...


@overload
def m_sequence(
    degree: int,
    poly: PolyLike | None = None,
    index: int = 1,
    output: Literal["bipolar"] = "decimal",
) -> npt.NDArray[np.float64]:
    ...


@export
def m_sequence(
    degree: Any,
    poly: Any = None,
    index: Any = 1,
    output: Any = "decimal",
) -> Any:
    r"""
    Generates a maximal-length sequence (m-sequence) from a Fibonacci linear feedback shift register (LFSR).

    Arguments:
        degree: The degree $n$ of the LFSR.
        poly: The characteristic polynomial of the LFSR over $\mathrm{GF}(q)$. The default is `None`, which uses the
            primitive polynomial of degree $n$ over $\mathrm{GF}(2)$, `galois.primitive_poly(2, degree)`.
        index: The index $i$ in $[1, q^{n})$ of the m-sequence. The index represents the initial state of the LFSR.
            The index dictates the phase of the m-sequence. The integer index is interpreted as a polynomial over
            $\mathrm{GF}(q)$, whose coefficients are the shift register values. The default is 1, which corresponds
            to the $[0, \dots, 0, 1]$ state.
        output: The output format of the m-sequence.

            - `"decimal"` (default): The m-sequence with decimal values in $[0, q^n)$.
            - `"field"`: The m-sequence as a Galois field array over $\mathrm{GF}(q^n)$.
            - `"bipolar"`: The m-sequence with bipolar values of 1 and -1. Only valid for $q = 2$.

    Returns:
        The length-$q^n - 1$ m-sequence.

    References:
        - https://en.wikipedia.org/wiki/Maximum_length_sequence

    Examples:
        Generate a maximal-length sequence of degree-4 over $\mathrm{GF}(2)$. Compare the sequence with index 1 to
        the sequence with index 2. They are just phase shifts of each other.

        .. ipython:: python

            sdr.m_sequence(4)
            sdr.m_sequence(4, index=2)

        Generate a maximal-length sequence of degree-4 over $\mathrm{GF}(3^2)$.

        .. ipython:: python

            c = galois.primitive_poly(3**2, 4); c
            x = sdr.m_sequence(4, poly=c); x
            x.size

    Group:
        sequences-maximal-length
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
        raise ValueError(f"Argument 'index' must be in [1, q^n), not {index}.")
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
