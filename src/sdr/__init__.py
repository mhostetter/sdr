"""
A Python package for software-defined radio (SDR) applications.
"""
try:
    from ._version import __version__, __version_tuple__
except ModuleNotFoundError:  # pragma: no cover
    import warnings

    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)
    warnings.warn(
        "An error occurred during package install where setuptools_scm failed to create a _version.py file. "
        "Defaulting version to 0.0.0."
    )

from . import plot
from ._conversion import *
from ._data import *
from ._detection import *
from ._farrow import *
from ._filter import *
from ._link_budget import *
from ._loop_filter import *
from ._measurement import *
from ._modulation import *
from ._nco import *
from ._pll import *
from ._probability import *
from ._sequence import *
from ._signal import *
from ._simulation import *
