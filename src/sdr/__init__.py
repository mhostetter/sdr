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

from ._farrow import FarrowResampler
from ._iir_filter import IIR
from ._nco import DDS, NCO
