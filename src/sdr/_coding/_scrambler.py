"""
A module containing various scramblers.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from galois.typing import ArrayLike, PolyLike

from .._helper import export
from .._sequence import FLFSR


@export
class AdditiveScrambler:
    r"""
    Implements an additive scrambler.

    Notes:
        .. code-block:: text
           :caption: Additive Scrambler Block Diagram

                    +--------------@<-------------@<------------@<-------------+
                    |              ^              ^             ^              |
                    |              | c_n-1        | c_n-2       | c_1          | c_0
                    |              | T[0]         | T[1]        | T[n-2]       | T[n-1]
                    |              |              |             |              |
                    |  +--------+  |  +--------+  |             |  +--------+  |
                    +->|  S[0]  |--+->|  S[1]  |--+---  ...  ---+->| S[n-1] |--+
                    |  +--------+     +--------+                   +--------+
                    |
                    v
            x[n] -->@--> y[n]

            S[k] = State vector
            T[k] = Taps vector
            x[n] = Input sequence
            y[n] = Output sequence
            @ = Finite field adder

    References:
        - https://en.wikipedia.org/wiki/Scrambler

    Examples:
        Construct the additive scrambler used in IEEE 802.11.

        .. ipython:: python

            c = galois.Poly.Degrees([7, 3, 0]); c
            scrambler = sdr.AdditiveScrambler(c)

        Scramble and descramble a sequence.

        .. ipython:: python

            x = np.random.randint(0, 2, 20); x
            y = scrambler.scramble(x); y
            xx = scrambler.descramble(y); xx
            np.array_equal(x, xx)

    Group:
        coding-scramblers
    """

    def __init__(
        self,
        characteristic_poly: PolyLike | None = None,
        feedback_poly: PolyLike | None = None,
        state: ArrayLike | None = None,
    ):
        r"""
        Creates an additive scrambler.

        Arguments:
            characteristic_poly: The characteristic polynomial
                $c(x) = x^{n} - c_{n-1} \cdot x^{n-1} - c_{n-2} \cdot x^{n-2} - \dots - c_{1} \cdot x - c_{0}$.
            feedback_poly: The feedback polynomial
                $f(x) = -c_{0} \cdot x^{n} - c_{1} \cdot x^{n-1} - \dots - c_{n-2} \cdot x^{2} - c_{n-1} \cdot x + 1$.

                .. note::
                    Either `characteristic_poly` or `feedback_poly` must be specified, but not both.

            state: The initial state vector $S = [S_0, S_1, \dots, S_{n-2}, S_{n-1}]$. The default is `None`
                which corresponds to all ones.

        See Also:
            FLFSR, galois.primitive_poly
        """
        self._lfsr = FLFSR(characteristic_poly, feedback_poly, state=state)

    def scramble(self, x: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        r"""
        Scrambles the input sequence $x[n]$.

        Arguments:
            x: The input sequence $x[n]$.

        Returns:
            The scrambled output sequence $y[n]$.
        """
        self.lfsr.reset()  # Set the initial state
        shift = self.lfsr.characteristic_poly.degree
        seq = self.lfsr.step(x.size + shift)
        seq = seq.view(np.ndarray)
        y = np.bitwise_xor(x, seq[shift:])
        return y

    def descramble(self, y: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        r"""
        Descrambles the input sequence $y[n]$.

        Arguments:
            y: The input sequence $y[n]$.

        Returns:
            The descrambled output sequence $x[n]$.
        """
        return self.scramble(y)

    @property
    def lfsr(self) -> FLFSR:
        r"""
        The Fibonacci LFSR used for scrambling.
        """
        return self._lfsr
