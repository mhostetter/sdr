"""
Conversions between decibel, linear, and signal-to-noise ratio representations.

This package provides functions for converting between:
- Linear and decibel (dB) scales for value, power, and voltage quantities
- Eb/N0, Es/N0, and S/N signal-to-noise ratio representations
"""

from __future__ import annotations

from ._decibels import *
from ._snr import *
