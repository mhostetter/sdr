"""
A module containing various helper functions for the library.
"""
import builtins
import sys

SPHINX_BUILD = hasattr(builtins, "__sphinx_build__")


def export(obj):
    """
    Marks an object for exporting into the public API.

    This decorator appends the object's name to the private module's __all__ list. The private module should
    then be imported in sdr/__init__.py using from ._private_module import *. It also modifies the object's
    __module__ to "sdr".
    """
    # Determine the private module that defined the object
    module = sys.modules[obj.__module__]

    if not SPHINX_BUILD:
        # Set the object's module to the first non-private module. This way the REPL will display the object
        # as sdr.obj and not sdr._private_module.obj.
        idx = obj.__module__.find("._")
        obj.__module__ = obj.__module__[:idx]

    # Append this object to the private module's "all" list
    public_members = getattr(module, "__all__", [])
    public_members.append(obj.__name__)
    setattr(module, "__all__", public_members)

    return obj
