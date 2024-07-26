"""
A module containing various correlation sequences.
"""

from __future__ import annotations

import math
from typing import Any, overload

import galois
import numpy as np
import numpy.typing as npt
import scipy.linalg
from galois import Poly
from galois.typing import PolyLike
from typing_extensions import Literal

from .._data import pack, unpack
from .._helper import export, verify_bool, verify_coprime, verify_isinstance, verify_literal, verify_scalar
from ._conversion import code_to_field, code_to_sequence, sequence_to_code
from ._maximum import is_preferred_pair, m_sequence, preferred_pairs


@overload
def barker_code(length: int, output: Literal["binary"] = "binary") -> npt.NDArray[np.int_]: ...


@overload
def barker_code(length: int, output: Literal["field"]) -> galois.FieldArray: ...


@overload
def barker_code(length: int, output: Literal["bipolar"]) -> npt.NDArray[np.float64]: ...


@export
def barker_code(length: Any, output: Any = "binary") -> Any:
    r"""
    Generates the Barker code/sequence of length $n$.

    Arguments:
        length: The length $n$ of the Barker code/sequence.
        output: The output format of the Barker code/sequence.

            - `"binary"` (default): The Barker code with binary values of 0 and 1.
            - `"field"`: The Barker code as a Galois field array over $\mathrm{GF}(2)$.
            - `"bipolar"`: The Barker sequence with bipolar values of 1 and -1.

    Returns:
        The Barker code/sequence of length $n$.

    Notes:
        Barker codes only exist for length $n = 1, 2, 3, 4, 5, 7, 11, 13$.

    Examples:
        Create a Barker code and sequence of length 13.

        .. ipython:: python

            sdr.barker_code(13)
            sdr.barker_code(13, output="bipolar")
            sdr.barker_code(13, output="field")

        Barker sequences have ideally minimal auto-correlation sidelobes of +1 or -1.

        .. ipython:: python

            x = sdr.barker_code(13, output="bipolar")

            @savefig sdr_barker_code_1.png
            plt.figure(); \
            sdr.plot.correlation(x, x, mode="circular");

    Group:
        sequences-correlation
    """
    verify_scalar(length, int=True)
    verify_literal(output, ["binary", "field", "bipolar"])

    if length == 1:
        code = np.array([1])
    elif length == 2:
        code = np.array([1, 0])
    elif length == 3:
        code = np.array([1, 1, 0])
    elif length == 4:
        code = np.array([1, 1, 0, 1])
    elif length == 5:
        code = np.array([1, 1, 1, 0, 1])
    elif length == 7:
        code = np.array([1, 1, 1, 0, 0, 1, 0])
    elif length == 11:
        code = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0])
    elif length == 13:
        code = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    else:
        raise ValueError(f"Barker sequence of length {length} does not exist.")

    if output == "binary":
        return code
    elif output == "field":
        return code_to_field(code)
    else:
        return code_to_sequence(code)


@overload
def hadamard_code(length: int, index: int, output: Literal["binary"] = "binary") -> npt.NDArray[np.int_]: ...


@overload
def hadamard_code(length: int, index: int, output: Literal["field"]) -> galois.FieldArray: ...


@overload
def hadamard_code(length: int, index: int, output: Literal["bipolar"]) -> npt.NDArray[np.float64]: ...


@export
def hadamard_code(length: Any, index: Any, output: Any = "binary") -> Any:
    r"""
    Generates the Hadamard code/sequence of length $n = 2^m$.

    Arguments:
        length: The length $n = 2^m$ of the Hadamard code/sequence.
        index: The index $i$ in $[0, n)$ of the Hadamard code.
        output: The output format of the Hadamard code/sequence.

            - `"binary"` (default): The Hadamard code with binary values of 0 and 1.
            - `"field"`: The Hadamard code as a Galois field array over $\mathrm{GF}(2)$.
            - `"bipolar"`: The Hadamard sequence with bipolar values of 1 and -1.

    Returns:
        The Hadamard code/sequence of length $n = 2^m$ and index $i$.

    References:
        - https://www.mathworks.com/help/comm/ref/comm.hadamardcode-system-object.html

    Examples:
        Create a Hadamard code and sequence of length 16.

        .. ipython:: python

            sdr.hadamard_code(16, 4)
            sdr.hadamard_code(16, 4, output="bipolar")
            sdr.hadamard_code(16, 4, output="field")

        The Hadamard and Walsh code sets are equivalent, however they are indexed differently.

        .. ipython:: python

            np.array_equal(sdr.hadamard_code(16, 3), sdr.walsh_code(16, 8))
            np.array_equal(sdr.hadamard_code(16, 11), sdr.walsh_code(16, 9))

        Hadamard sequences have zero cross-correlation when time aligned.

        .. ipython:: python

            x1 = sdr.hadamard_code(32, 30, output="bipolar"); \
            x2 = sdr.hadamard_code(32, 18, output="bipolar"); \
            x3 = sdr.hadamard_code(32, 27, output="bipolar");

            @savefig sdr_hadamard_code_1.png
            plt.figure(); \
            sdr.plot.time_domain(x1 + 3); \
            sdr.plot.time_domain(x2 + 0); \
            sdr.plot.time_domain(x3 - 3)

        Hadamard sequence auto-correlation sidelobes are not uniform as a function of sequence index.
        In fact, the sidelobes can be quite high.

        .. ipython:: python

            @savefig sdr_hadamard_code_2.png
            plt.figure(); \
            sdr.plot.correlation(x1, x1, mode="circular"); \
            sdr.plot.correlation(x2, x2, mode="circular"); \
            sdr.plot.correlation(x3, x3, mode="circular"); \
            plt.ylim(0, 32);

        Hadamard sequences have zero cross-correlation when time aligned. However, the sidelobes can be quite
        large when time misaligned. Because of this, Hadamard sequences for spreading codes are useful only when
        precise time information is known.

        .. ipython:: python

            @savefig sdr_hadamard_code_3.png
            plt.figure(); \
            sdr.plot.correlation(x1, x2, mode="circular"); \
            sdr.plot.correlation(x1, x3, mode="circular"); \
            sdr.plot.correlation(x2, x3, mode="circular"); \
            plt.ylim(0, 32);

    Group:
        sequences-correlation
    """
    verify_scalar(length, int=True, inclusive_min=2, power_of_two=True)
    verify_scalar(index, int=True, inclusive_min=0, exclusive_max=length)
    verify_literal(output, ["binary", "field", "bipolar"])

    H = scipy.linalg.hadamard(length)
    sequence = H[index]
    sequence = sequence.astype(float)

    if output == "binary":
        return sequence_to_code(sequence)
    elif output == "field":
        return code_to_field(sequence_to_code(sequence))
    else:
        return sequence


@overload
def walsh_code(length: int, index: int, output: Literal["binary"] = "binary") -> npt.NDArray[np.int_]: ...


@overload
def walsh_code(length: int, index: int, output: Literal["field"]) -> galois.FieldArray: ...


@overload
def walsh_code(length: int, index: int, output: Literal["bipolar"]) -> npt.NDArray[np.float64]: ...


@export
def walsh_code(length: Any, index: Any, output: Any = "binary") -> Any:
    r"""
    Generates the Walsh code/sequence of length $n = 2^m$.

    Arguments:
        length: The length $n = 2^m$ of the Walsh code/sequence.
        index: The index $i$ in $[0, n)$ of the Walsh code. Indicates how many transitions there are in the code.
        output: The output format of the Walsh code/sequence.

            - `"binary"` (default): The Walsh code with binary values of 0 and 1.
            - `"field"`: The Walsh code as a Galois field array over $\mathrm{GF}(2)$.
            - `"bipolar"`: The Walsh sequence with bipolar values of 1 and -1.

    Returns:
        The Walsh code/sequence of length $n = 2^m$ and index $i$.

    References:
        - https://www.mathworks.com/help/comm/ref/comm.walshcode-system-object.html

    Examples:
        Create a Walsh code and sequence of length 16. The code index 4 indicates that there are 4 transitions
        in the code.

        .. ipython:: python

            sdr.walsh_code(16, 4)
            sdr.walsh_code(16, 4, output="bipolar")
            sdr.walsh_code(16, 4, output="field")

        The Hadamard and Walsh code sets are equivalent, however they are indexed differently.

        .. ipython:: python

            np.array_equal(sdr.hadamard_code(16, 3), sdr.walsh_code(16, 8))
            np.array_equal(sdr.hadamard_code(16, 11), sdr.walsh_code(16, 9))

        Walsh sequences have zero cross-correlation when time aligned.

        .. ipython:: python

            x1 = sdr.walsh_code(32, 10, output="bipolar"); \
            x2 = sdr.walsh_code(32, 14, output="bipolar"); \
            x3 = sdr.walsh_code(32, 18, output="bipolar");

            @savefig sdr_walsh_code_1.png
            plt.figure(); \
            sdr.plot.time_domain(x1 + 3); \
            sdr.plot.time_domain(x2 + 0); \
            sdr.plot.time_domain(x3 - 3)

        Walsh sequence auto-correlation sidelobes are not uniform as a function of sequence index.
        In fact, the sidelobes can be quite high.

        .. ipython:: python

            @savefig sdr_walsh_code_2.png
            plt.figure(); \
            sdr.plot.correlation(x1, x1, mode="circular"); \
            sdr.plot.correlation(x2, x2, mode="circular"); \
            sdr.plot.correlation(x3, x3, mode="circular"); \
            plt.ylim(0, 32);

        Walsh sequences have zero cross-correlation when time aligned. However, the sidelobes can be quite
        large when time misaligned. Because of this, Walsh sequences for spreading codes are useful only when
        precise time information is known.

        .. ipython:: python

            @savefig sdr_walsh_code_3.png
            plt.figure(); \
            sdr.plot.correlation(x1, x2, mode="circular"); \
            sdr.plot.correlation(x1, x3, mode="circular"); \
            sdr.plot.correlation(x2, x3, mode="circular"); \
            plt.ylim(0, 32);

    Group:
        sequences-correlation
    """
    verify_scalar(length, int=True, inclusive_min=2, power_of_two=True)
    verify_scalar(index, int=True, inclusive_min=0, exclusive_max=length)
    verify_literal(output, ["binary", "field", "bipolar"])

    H = scipy.linalg.hadamard(length)

    # Find the index of the Hadamard matrix that corresponds to `index` transitions
    bits = int(math.log2(length))
    index_bits = unpack(index, bits)
    h_index_bits = np.concatenate(([index_bits[0]], np.bitwise_xor(index_bits[:-1], index_bits[1:])))[::-1]
    h_index = pack(h_index_bits, bits)[0]

    sequence = H[h_index]
    sequence = sequence.astype(float)

    if output == "binary":
        return sequence_to_code(sequence)
    elif output == "field":
        return code_to_field(sequence_to_code(sequence))
    else:
        return sequence


@overload
def gold_code(
    length: int,
    index: int = 0,
    poly1: PolyLike | None = None,
    poly2: PolyLike | None = None,
    verify: bool = True,
    output: Literal["binary"] = "binary",
) -> npt.NDArray[np.int_]: ...


@overload
def gold_code(
    length: int,
    index: int = 0,
    poly1: PolyLike | None = None,
    poly2: PolyLike | None = None,
    verify: bool = True,
    output: Literal["field"] = "binary",
) -> galois.FieldArray: ...


@overload
def gold_code(
    length: int,
    index: int = 0,
    poly1: PolyLike | None = None,
    poly2: PolyLike | None = None,
    verify: bool = True,
    output: Literal["bipolar"] = "binary",
) -> npt.NDArray[np.float64]: ...


@export
def gold_code(
    length: Any, index: Any = 0, poly1: Any = None, poly2: Any = None, verify: Any = True, output: Any = "binary"
) -> Any:
    r"""
    Generates the Gold code/sequence of length $n = 2^m - 1$.

    Arguments:
        length: The length $n = 2^m - 1$ of the Gold code/sequence.
        index: The index $i$ in $[-2, n)$ of the Gold code.
        poly1: The primitive polynomial of degree $m$ over $\mathrm{GF}(2)$ for the first $m$-sequence. If `None`,
            a preferred pair is found using :func:`sdr.preferred_pairs`.
        poly2: The primitive polynomial of degree $m$ over $\mathrm{GF}(2)$ for the second $m$-sequence. If `None`,
            a preferred pair is found using :func:`sdr.preferred_pairs`.
        verify: Indicates whether to verify that the provided polynomials are a preferred pair using
            :func:`sdr.is_preferred_pair`.
        output: The output format of the Gold code/sequence.

            - `"binary"` (default): The Gold code with binary values of 0 and 1.
            - `"field"`: The Gold code as a Galois field array over $\mathrm{GF}(2)$.
            - `"bipolar"`: The Gold sequence with bipolar values of 1 and -1.

    Returns:
        The Gold code/sequence of length $n = 2^m - 1$ and index $i$.

    See Also:
        sdr.preferred_pairs, sdr.is_preferred_pair

    Notes:
        Gold codes are generated by combining two preferred pair $m$-sequences, $u$ and $v$, using the formula

        $$
        c = \begin{cases}
        u & \text{if $i = -2$} \\
        v & \text{if $i = -1$} \\
        u \oplus T^i v & \text{otherwise} ,
        \end{cases}
        $$

        where $i$ is the code index, $\oplus$ is addition in $\mathrm{GF}(2)$ (or the XOR operation), and $T^i$ is a
        left shift by $i$ positions.

        Gold codes are PN sequence with good auto-correlation and cross-correlation properties. The Gold code set
        contains $2^m + 1$ sequences of length $2^m - 1$. The correlation sides are guaranteed to be less than
        or equal to $t(m)$.

        $$
        t(m) = \begin{cases}
        2^{(m+1)/2} + 1 & \text{if $m$ is odd} \\
        2^{(m+2)/2} + 1 & \text{if $m$ is even}
        \end{cases}
        $$

        There are no preferred pairs with degree $m$ divisible by 4. Therefore, there are no Gold codes with degree
        $m$ divisible by 4.

    References:
        - John Proakis, *Digital Communications*, Chapter 12.2-5: Generation of PN Sequences.

    Examples:
        Create a Gold code and sequence of length 7.

        .. ipython:: python

            sdr.gold_code(7, 1)
            sdr.gold_code(7, 1, output="bipolar")
            sdr.gold_code(7, 1, output="field")

        Create several Gold codes of length 63.

        .. ipython:: python

            x1 = sdr.gold_code(63, 0, output="bipolar"); \
            x2 = sdr.gold_code(63, 1, output="bipolar"); \
            x3 = sdr.gold_code(63, 2, output="bipolar");

            @savefig sdr_gold_code_1.png
            plt.figure(); \
            sdr.plot.time_domain(x1 + 3); \
            sdr.plot.time_domain(x2 + 0); \
            sdr.plot.time_domain(x3 - 3)

        Examine the auto-correlation of the Gold sequences.

        .. ipython:: python

            @savefig sdr_gold_code_2.png
            plt.figure(); \
            sdr.plot.correlation(x1, x1, mode="circular"); \
            sdr.plot.correlation(x2, x2, mode="circular"); \
            sdr.plot.correlation(x3, x3, mode="circular"); \
            plt.ylim(0, 63);

        Examine the cross-correlation of the Gold sequences.

        .. ipython:: python

            @savefig sdr_gold_code_3.png
            plt.figure(); \
            sdr.plot.correlation(x1, x2, mode="circular"); \
            sdr.plot.correlation(x1, x3, mode="circular"); \
            sdr.plot.correlation(x2, x3, mode="circular"); \
            plt.ylim(0, 63);

    Group:
        sequences-correlation
    """
    verify_scalar(length, int=True, positive=True)
    verify_scalar(length + 1, power_of_two=True)
    m = int(np.log2(length + 1))
    verify_scalar(index, int=True, inclusive_min=-2, exclusive_max=length)
    verify_bool(verify)
    verify_literal(output, ["binary", "field", "bipolar"])

    if poly1 is None and poly2 is None:
        poly1, poly2 = next(preferred_pairs(m))
    elif poly1 is None:
        poly1 = next(preferred_pairs(m, poly2))[1]
    elif poly2 is None:
        poly2 = next(preferred_pairs(m, poly1))[1]
    else:
        if verify and not is_preferred_pair(poly1, poly2):
            raise ValueError(
                f"Arguments 'poly1' and 'poly2' must be a preferred pair to generate a Gold code, {poly1} and {poly2} are not."
                + " You can pass 'verify=False' to disable this check."
            )

    u = m_sequence(m, poly=poly1, index=1, output="decimal")
    v = m_sequence(m, poly=poly2, index=1, output="decimal")

    if index == -2:
        code = u
    elif index == -1:
        code = v
    else:
        code = np.bitwise_xor(u, np.roll(v, -index))

    if output == "binary":
        return code
    elif output == "field":
        return code_to_field(code)
    else:
        return code_to_sequence(code)


@overload
def kasami_code(
    length: int, index: int | tuple[int, int] = 0, poly: PolyLike | None = None, output: Literal["binary"] = "binary"
) -> npt.NDArray[np.int_]: ...


@overload
def kasami_code(
    length: int, index: int | tuple[int, int] = 0, poly: PolyLike | None = None, output: Literal["field"] = "binary"
) -> galois.FieldArray: ...


@overload
def kasami_code(
    length: int, index: int | tuple[int, int] = 0, poly: PolyLike | None = None, output: Literal["bipolar"] = "binary"
) -> npt.NDArray[np.float64]: ...


@export
def kasami_code(length: Any, index: Any = 0, poly: Any = None, output: Any = "binary") -> Any:
    r"""
    Generates the Kasami code/sequence of length $n = 2^m - 1$.

    Arguments:
        length: The length $n = 2^m - 1$ of the Kasami code/sequence. The degree $m$ must be even.
        index: The index of the Kasami code.

            - `int`: The index $j$ in $[-1, 2^{m/2} - 1)$ from the Kasami code small set. There are $2^{m/2}$ codes
              in the small set.
            - `tuple[int, int]`: The index $(i, j)$ from the Kasami code large set, with $i \in [-2, 2^m - 1)$ and
              $j \in [-1, 2^{m/2} - 1)$. There are $(2^m + 1) \cdot 2^{m/2}$ codes in the large set.

        poly: The primitive polynomial of degree $m$ over $\mathrm{GF}(2)$. The default is `None`, which uses the
            default primitive polynomial of degree $m$, i.e. `galois.primitive_poly(2, m)`.

        output: The output format of the Kasami code/sequence.

            - `"binary"` (default): The Kasami code with binary values of 0 and 1.
            - `"field"`: The Kasami code as a Galois field array over $\mathrm{GF}(2)$.
            - `"bipolar"`: The Kasami sequence with bipolar values of 1 and -1.

    Returns:
        The Kasami code/sequence of length $n = 2^m - 1$.

    References:
        - https://en.wikipedia.org/wiki/Kasami_code

    Examples:
        Create a Kasami code and sequence of length 15.

        .. ipython:: python

            sdr.kasami_code(15, 1)
            sdr.kasami_code(15, 1, output="bipolar")
            sdr.kasami_code(15, 1, output="field")

        Create several Kasami codes of length 63 from the small set.

        .. ipython:: python

            x1 = sdr.kasami_code(63, 0, output="bipolar"); \
            x2 = sdr.kasami_code(63, 1, output="bipolar"); \
            x3 = sdr.kasami_code(63, 2, output="bipolar");

            @savefig sdr_kasami_code_1.png
            plt.figure(); \
            sdr.plot.time_domain(x1 + 3); \
            sdr.plot.time_domain(x2 + 0); \
            sdr.plot.time_domain(x3 - 3)

        Examine the auto-correlation of the Kasami sequences.

        .. ipython:: python

            @savefig sdr_kasami_code_2.png
            plt.figure(); \
            sdr.plot.correlation(x1, x1, mode="circular"); \
            sdr.plot.correlation(x2, x2, mode="circular"); \
            sdr.plot.correlation(x3, x3, mode="circular"); \
            plt.ylim(0, 63);

        Examine the cross-correlation of the Kasami sequences.

        .. ipython:: python

            @savefig sdr_kasami_code_3.png
            plt.figure(); \
            sdr.plot.correlation(x1, x2, mode="circular"); \
            sdr.plot.correlation(x1, x3, mode="circular"); \
            sdr.plot.correlation(x2, x3, mode="circular"); \
            plt.ylim(0, 63);

    Group:
        sequences-correlation
    """
    verify_scalar(length, int=True, positive=True)
    verify_scalar(length + 1, power_of_two=True)
    n = int(np.log2(length + 1))
    verify_scalar(n, even=True)
    verify_isinstance(index, (int, tuple))

    if poly is None:
        c = galois.primitive_poly(2, n)
    else:
        c = Poly._PolyLike(poly, field=galois.GF(2))

    if isinstance(index, int):
        code = _kasami_small_set(n, c, index)
    elif isinstance(index, tuple):
        code = _kasami_large_set(n, c, index)

    if output == "binary":
        return code
    elif output == "field":
        return code_to_field(code)
    else:
        return code_to_sequence(code)


def _kasami_small_set(degree: int, poly: Poly, index: int) -> npt.NDArray[np.int_]:
    if not degree % 2 == 0:
        raise ValueError(f"Argument 'degree' must be even, not {degree}.")

    j = index
    if not -1 <= j < 2 ** (degree // 2) - 1:
        raise ValueError(f"Argument 'index' must be between -1 and {2**(degree//2) - 1}, not {j}.")

    u = m_sequence(degree, poly=poly, index=1, output="decimal")
    length = u.size

    stride = 2 ** (degree // 2) + 1
    idxs = (np.arange(0, length) * stride) % length
    w = u[idxs]

    if j == -1:
        code = u
    else:
        code = np.bitwise_xor(u, np.roll(w, -j))

    return code


def _kasami_large_set(degree: int, poly: Poly, index: tuple[int, int]) -> npt.NDArray[np.int_]:
    if not degree % 4 == 2:
        raise ValueError(f"Argument 'degree' must be 2 mod 4, not {degree}.")

    i, j = index
    if not -2 <= i < 2**degree - 1:
        raise ValueError(f"Argument 'index[0]' must be between -2 and {2**degree - 1}, not {i}.")
    if not -1 <= j < 2 ** (degree // 2) - 1:
        raise ValueError(f"Argument 'index[1]' must be between -1 and {2**(degree//2) - 1}, not {j}.")

    u = m_sequence(degree, poly=poly, index=1, output="decimal")
    length = u.size

    stride = 2 ** (degree // 2) + 1
    idxs = (np.arange(0, length) * stride) % length
    w = u[idxs]

    stride = 2 ** (degree // 2 + 1) + 1
    idxs = (np.arange(0, length) * stride) % length
    v = u[idxs]

    if j == -1:
        if i == -2:
            code = u
        elif i == -1:
            code = v
        else:
            code = np.bitwise_xor(u, np.roll(v, -i))
    else:
        if i == -2:
            code = np.bitwise_xor(u, np.roll(w, -j))
        elif i == -1:
            code = np.bitwise_xor(v, np.roll(w, -j))
        else:
            code = np.bitwise_xor(np.bitwise_xor(u, np.roll(v, -i)), np.roll(w, -j))

    return code


@export
def zadoff_chu_sequence(length: int, root: int, shift: int = 0) -> npt.NDArray[np.complex128]:
    r"""
    Generates the root-$u$ Zadoff-Chu sequence of length $N$.

    Arguments:
        length: The length $N$ of the Zadoff-Chu sequence.
        root: The root $0 < u < N$ of the Zadoff-Chu sequence. The root must be relatively prime to the length,
            i.e., $\gcd(u, N) = 1$.
        shift: The shift $q \in \mathbb{Z}$ of the Zadoff-Chu sequence. When $q \ne 0$, the returned sequence
            is a cyclic shift of the root-$u$ Zadoff-Chu sequence.

    Returns:
        The root-$u$ Zadoff-Chu sequence of length $N$.

    Notes:
        The root-$u$ Zadoff-Chu sequence with length $N$ and shift $q$ is defined as

        $$x_u[n] = \exp \left( -j \frac{\pi u n (n + c_{f} + 2q)}{N} \right) ,$$

        where $c_{f} = N \mod 2$.

    References:
        - https://en.wikipedia.org/wiki/Zadoff%E2%80%93Chu_sequence

    Examples:
        Create a root-3 Zadoff-Chu sequence $x_3[n]$ with length 139.

        .. ipython:: python

            N = 139
            x3 = sdr.zadoff_chu_sequence(N, 3)

            @savefig sdr_zadoff_chu_1.png
            plt.figure(); \
            sdr.plot.constellation(x3, linestyle="-", linewidth=0.5); \
            plt.title(f"Root-3 Zadoff-Chu sequence of length {N}");

        The *periodic* auto-correlation of a Zadoff-Chu sequence has sidelobes with magnitude 0.

        .. ipython:: python

            @savefig sdr_zadoff_chu_2.png
            plt.figure(); \
            sdr.plot.correlation(x3, x3, mode="circular"); \
            plt.ylim(0, N);

        Create a root-5 Zadoff-Chu sequence $x_5[n]$ with length 139.

        .. ipython:: python

            x5 = sdr.zadoff_chu_sequence(N, 5)

            @savefig sdr_zadoff_chu_3.png
            plt.figure(); \
            sdr.plot.constellation(x5, linestyle="-", linewidth=0.5); \
            plt.title(f"Root-5 Zadoff-Chu sequence of length {N}");

        The *periodic* cross-correlation of two prime-length Zadoff-Chu sequences with different roots has sidelobes
        with magnitude $1 / \sqrt{N}$.

        .. ipython:: python

            @savefig sdr_zadoff_chu_4.png
            plt.figure(); \
            sdr.plot.correlation(x3, x5, mode="circular"); \
            plt.ylim(0, N);

    Group:
        sequences-correlation
    """
    verify_scalar(length, int=True, inclusive_min=2)
    verify_scalar(root, int=True, exclusive_min=0, exclusive_max=length)
    verify_coprime(length, root)
    verify_scalar(shift, int=True)

    n = np.arange(length)
    cf = length % 2
    xu = np.exp(-1j * np.pi * root * n * (n + cf + 2 * shift) / length)

    return xu
