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
from typing_extensions import Literal

from .._data import pack, unpack
from .._helper import export
from ._conversion import code_to_field, code_to_sequence, sequence_to_code


@overload
def barker_code(length: int, output: Literal["binary"]) -> npt.NDArray[np.int_]:
    ...


@overload
def barker_code(length: int, output: Literal["field"]) -> galois.FieldArray:
    ...


@overload
def barker_code(length: int, output: Literal["bipolar"] = "bipolar") -> npt.NDArray[np.float_]:
    ...


@export
def barker_code(length: Any, output: Any = "bipolar") -> Any:
    r"""
    Returns the Barker code/sequence of length $N$.

    Arguments:
        length: The length $N$ of the Barker code/sequence.
        output: The output format of the Barker code/sequence.

            - `"binary"`: The Barker code with binary values of 0 and 1.
            - `"field"`: The Barker code as a Galois field array over $\mathrm{GF}(2)$.
            - `"bipolar"`: The Barker sequence with bipolar values of 1 and -1.

    Returns:
        The Barker code/sequence of length $N$.

    Examples:
        Create a Barker code and sequence of length 13.

        .. ipython:: python

            code = sdr.barker_code(13, output="binary"); code
            seq = sdr.barker_code(13); seq

        Barker sequences have ideally-minimal autocorrelation sidelobes of +1 or -1.

        .. ipython:: python

            corr = np.correlate(seq, seq, mode="full")
            lag = np.arange(-seq.size + 1, seq.size)

            @savefig sdr_barker_1.png
            plt.figure(); \
            sdr.plot.time_domain(lag, np.abs(corr)); \
            plt.xlabel("Lag"); \
            plt.title("Autocorrelation of length-13 Barker sequence");

    Group:
        sequences-correlation
    """
    if not isinstance(length, int):
        raise TypeError(f"Argument 'length' must be of type 'int', not {type(length)}.")
    if not output in ("binary", "field", "bipolar"):
        raise ValueError(f"Argument 'output' must be either 'binary', 'field', or 'bipolar', not {output}.")

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
def hadamard_code(length: int, index: int, output: Literal["binary"]) -> npt.NDArray[np.int_]:
    ...


@overload
def hadamard_code(length: int, index: int, output: Literal["field"]) -> galois.FieldArray:
    ...


@overload
def hadamard_code(length: int, index: int, output: Literal["bipolar"] = "bipolar") -> npt.NDArray[np.float_]:
    ...


@export
def hadamard_code(length: Any, index: Any, output: Any = "bipolar") -> Any:
    r"""
    Returns the Hadamard code/sequence of length $N$.

    Arguments:
        length: The length $N$ of the Hadamard code/sequence. Must be a power of 2.
        index: The index $i$ of the Hadamard code.
        output: The output format of the Hadamard code/sequence.

            - `"binary"`: The Hadamard code with binary values of 0 and 1.
            - `"field"`: The Hadamard code as a Galois field array over $\mathrm{GF}(2)$.
            - `"bipolar"`: The Hadamard sequence with bipolar values of 1 and -1.

    Returns:
        The Hadamard code/sequence of length $N$ and index $i$.

    References:
        - https://www.mathworks.com/help/comm/ref/comm.hadamardcode-system-object.html

    Examples:
        Create a Hadamard code and sequence of length 16.

        .. ipython:: python

            code = sdr.hadamard_code(16, 4, output="binary"); code
            seq = sdr.hadamard_code(16, 4); seq

        The Hadamard and Walsh code sets are equivalent, however they are indexed differently.

        .. ipython:: python

            np.array_equal(sdr.hadamard_code(16, 3), sdr.walsh_code(16, 8))
            np.array_equal(sdr.hadamard_code(16, 11), sdr.walsh_code(16, 9))

        Hadamard sequences have zero cross correlation when time aligned.

        .. ipython:: python

            seq1 = sdr.hadamard_code(16, 4); \
            seq2 = sdr.hadamard_code(16, 10); \
            seq3 = sdr.hadamard_code(16, 15);

            @savefig sdr_hadamard_1.png
            plt.figure(); \
            sdr.plot.time_domain(seq1 + 3, label="Index 4"); \
            sdr.plot.time_domain(seq2 + 0, label="Index 10"); \
            sdr.plot.time_domain(seq3 - 3, label="Index 15")

        Hadamard sequences have zero cross correlation when time aligned. However, the sidelobes can be quite
        large when time misaligned. Because of this, Hadamard sequences for spreading codes are useful only when
        precise time information is known.

        .. ipython:: python

            lag = np.arange(-seq1.size + 1, seq1.size); \
            xcorr12 = np.correlate(seq1, seq2, mode="full"); \
            xcorr13 = np.correlate(seq1, seq3, mode="full"); \
            xcorr23 = np.correlate(seq2, seq3, mode="full");

            @savefig sdr_hadamard_2.png
            plt.figure(); \
            sdr.plot.time_domain(lag, np.abs(xcorr12), label="4 and 10"); \
            sdr.plot.time_domain(lag, np.abs(xcorr13), label="4 and 15"); \
            sdr.plot.time_domain(lag, np.abs(xcorr23), label="10 and 15"); \
            plt.xlabel("Lag"); \
            plt.title("Cross correlation of length-16 Hadamard sequences");

        Hadamard sequence autocorrelation sidelobes are not uniform as a function of sequence index.
        In fact, the sidelobes can be quite high.

        .. ipython:: python

            lag = np.arange(-seq1.size + 1, seq1.size); \
            acorr1 = np.correlate(seq1, seq1, mode="full"); \
            acorr2 = np.correlate(seq2, seq2, mode="full"); \
            acorr3 = np.correlate(seq3, seq3, mode="full");

            @savefig sdr_hadamard_3.png
            plt.figure(); \
            sdr.plot.time_domain(lag, np.abs(acorr1), label="Index 4"); \
            sdr.plot.time_domain(lag, np.abs(acorr2), label="Index 10"); \
            sdr.plot.time_domain(lag, np.abs(acorr3), label="Index 15"); \
            plt.xlabel("Lag"); \
            plt.title("Autocorrelation of length-16 Hadamard sequences");

    Group:
        sequences-correlation
    """
    if not isinstance(length, int):
        raise TypeError(f"Argument 'length' must be an integer, not {type(length).__name__}.")
    if not length >= 2:
        raise ValueError(f"Argument 'length' must be greater than or equal to 1, not {length}.")
    if not length & (length - 1) == 0:
        raise ValueError(f"Argument 'length' must be a power of 2, not {length}.")

    if not isinstance(index, int):
        raise TypeError(f"Argument 'index' must be an integer, not {type(index).__name__}.")
    if not 0 <= index < length:
        raise ValueError(f"Argument 'index' must be between 0 and {length - 1}, not {index}.")

    if not output in ("binary", "field", "bipolar"):
        raise ValueError(f"Argument 'output' must be either 'binary', 'field', or 'bipolar', not {output}.")

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
def walsh_code(length: int, index: int, output: Literal["binary"]) -> npt.NDArray[np.int_]:
    ...


@overload
def walsh_code(length: int, index: int, output: Literal["field"]) -> galois.FieldArray:
    ...


@overload
def walsh_code(length: int, index: int, output: Literal["bipolar"] = "bipolar") -> npt.NDArray[np.float_]:
    ...


@export
def walsh_code(length: Any, index: Any, output: Any = "bipolar") -> Any:
    r"""
    Returns the Walsh code/sequence of length $N$.

    Arguments:
        length: The length $N$ of the Walsh code/sequence. Must be a power of 2.
        index: The index $i$ of the Walsh code. Indicates how many transitions there are in the code.
        output: The output format of the Walsh code/sequence.

            - `"binary"`: The Walsh code with binary values of 0 and 1.
            - `"field"`: The Walsh code as a Galois field array over $\mathrm{GF}(2)$.
            - `"bipolar"`: The Walsh sequence with bipolar values of 1 and -1.

    Returns:
        The Walsh code/sequence of length $N$ and index $i$.

    References:
        - https://www.mathworks.com/help/comm/ref/comm.walshcode-system-object.html

    Examples:
        Create a Walsh code and sequence of length 16. The code index 4 indicates that there are 4 transitions
        in the code.

        .. ipython:: python

            code = sdr.walsh_code(16, 4, output="binary"); code
            seq = sdr.walsh_code(16, 4); seq

        The Hadamard and Walsh code sets are equivalent, however they are indexed differently.

        .. ipython:: python

            np.array_equal(sdr.hadamard_code(16, 3), sdr.walsh_code(16, 8))
            np.array_equal(sdr.hadamard_code(16, 11), sdr.walsh_code(16, 9))

        Walsh sequences have zero cross correlation when time aligned.

        .. ipython:: python

            seq1 = sdr.walsh_code(16, 4); \
            seq2 = sdr.walsh_code(16, 10); \
            seq3 = sdr.walsh_code(16, 15);

            @savefig sdr_walsh_1.png
            plt.figure(); \
            sdr.plot.time_domain(seq1 + 3, label="Index 4"); \
            sdr.plot.time_domain(seq2 + 0, label="Index 10"); \
            sdr.plot.time_domain(seq3 - 3, label="Index 15")

        Walsh sequences have zero cross correlation when time aligned. However, the sidelobes can be quite
        large when time misaligned. Because of this, Walsh sequences for spreading codes are useful only when
        precise time information is known.

        .. ipython:: python

            lag = np.arange(-seq1.size + 1, seq1.size); \
            xcorr12 = np.correlate(seq1, seq2, mode="full"); \
            xcorr13 = np.correlate(seq1, seq3, mode="full"); \
            xcorr23 = np.correlate(seq2, seq3, mode="full");

            @savefig sdr_walsh_2.png
            plt.figure(); \
            sdr.plot.time_domain(lag, np.abs(xcorr12), label="4 and 10"); \
            sdr.plot.time_domain(lag, np.abs(xcorr13), label="4 and 15"); \
            sdr.plot.time_domain(lag, np.abs(xcorr23), label="10 and 15"); \
            plt.xlabel("Lag"); \
            plt.title("Cross correlation of length-16 Walsh sequences");

        Walsh sequence autocorrelation sidelobes are not uniform as a function of sequence index.
        In fact, the sidelobes can be quite high.

        .. ipython:: python

            lag = np.arange(-seq1.size + 1, seq1.size); \
            acorr1 = np.correlate(seq1, seq1, mode="full"); \
            acorr2 = np.correlate(seq2, seq2, mode="full"); \
            acorr3 = np.correlate(seq3, seq3, mode="full");

            @savefig sdr_walsh_3.png
            plt.figure(); \
            sdr.plot.time_domain(lag, np.abs(acorr1), label="Index 4"); \
            sdr.plot.time_domain(lag, np.abs(acorr2), label="Index 10"); \
            sdr.plot.time_domain(lag, np.abs(acorr3), label="Index 15"); \
            plt.xlabel("Lag"); \
            plt.title("Autocorrelation of length-16 Walsh sequences");

    Group:
        sequences-correlation
    """
    if not isinstance(length, int):
        raise TypeError(f"Argument 'length' must be an integer, not {type(length).__name__}.")
    if not length >= 2:
        raise ValueError(f"Argument 'length' must be greater than or equal to 1, not {length}.")
    if not length & (length - 1) == 0:
        raise ValueError(f"Argument 'length' must be a power of 2, not {length}.")

    if not isinstance(index, int):
        raise TypeError(f"Argument 'index' must be an integer, not {type(index).__name__}.")
    if not 0 <= index < length:
        raise ValueError(f"Argument 'index' must be between 0 and {length - 1}, not {index}.")

    if not output in ("binary", "field", "bipolar"):
        raise ValueError(f"Argument 'output' must be either 'binary', 'field', or 'bipolar', not {output}.")

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


@export
def zadoff_chu_sequence(length: int, root: int, shift: int = 0) -> npt.NDArray[np.complex_]:
    r"""
    Returns the root-$u$ Zadoff-Chu sequence of length $N$.

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

        The *periodic* autocorrelation of a Zadoff-Chu sequence has sidelobes of magnitude 0.

        .. ipython:: python

            # Perform periodic autocorrelation
            corr = np.correlate(np.roll(np.tile(x3, 2), -N//2), x3, mode="valid")
            lag = np.arange(-N//2 + 1, N//2 + 2)

            @savefig sdr_zadoff_chu_2.png
            plt.figure(); \
            sdr.plot.time_domain(lag, np.abs(corr) / N); \
            plt.ylim(0, 1); \
            plt.xlabel("Lag"); \
            plt.title(f"Periodic autocorrelation of root-3 Zadoff-Chu sequence of length {N}");

        Create a root-5 Zadoff-Chu sequence $x_5[n]$ with length 139.

        .. ipython:: python

            x5 = sdr.zadoff_chu_sequence(N, 5)

            @savefig sdr_zadoff_chu_3.png
            plt.figure(); \
            sdr.plot.constellation(x5, linestyle="-", linewidth=0.5); \
            plt.title(f"Root-5 Zadoff-Chu sequence of length {N}");

        The *periodic* cross correlation of two prime-length Zadoff-Chu sequences with different roots has sidelobes
        with magnitude $1 / \sqrt{N}$.

        .. ipython:: python

            # Perform periodic cross correlation
            xcorr = np.correlate(np.roll(np.tile(x3, 2), -N//2), x5, mode="valid")
            lag = np.arange(-N//2 + 1, N//2 + 2)

            @savefig sdr_zadoff_chu_4.png
            plt.figure(); \
            sdr.plot.time_domain(lag, np.abs(xcorr) / N); \
            plt.ylim(0, 1); \
            plt.xlabel("Lag"); \
            plt.title(f"Periodic cross correlation of root-3 and root-5 Zadoff-Chu sequences of length {N}");

    Group:
        sequences-correlation
    """
    if not isinstance(length, int):
        raise TypeError(f"Argument 'length' must be of type 'int', not {type(length)}.")
    if not length > 1:
        raise ValueError(f"Argument 'length' must be greater than 1, not {length}.")

    if not isinstance(root, int):
        raise TypeError(f"Argument 'root' must be of type 'int', not {type(root)}.")
    if not 0 < root < length:
        raise ValueError(f"Argument 'root' must be greater than 0 and less than 'length', not {root}.")
    if not math.gcd(length, root) == 1:
        raise ValueError(f"Argument 'root' must be relatively prime to 'length'; {root} and {length} are not.")

    if not isinstance(shift, int):
        raise TypeError(f"Argument 'shift' must be of type 'int', not {type(shift)}.")

    n = np.arange(length)
    cf = length % 2
    xu = np.exp(-1j * np.pi * root * n * (n + cf + 2 * shift) / length)

    return xu
