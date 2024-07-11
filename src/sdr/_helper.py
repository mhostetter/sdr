"""
A module containing various helper functions for the library.
"""

import builtins
import inspect
import sys
from typing import Any

import numpy as np
import numpy.typing as npt

SPHINX_BUILD = hasattr(builtins, "__sphinx_build__")


def _argument_names():
    """
    Finds the source code argument names from the function that called a verification function.
    """
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[2]  # function() -> verify() -> _argument_name()
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    args = string[string.find("(") + 1 : -1].split(",")
    args = [arg.strip() for arg in args]  # Strip leading/trailing whitespace
    # args = [arg.split("=")[0].strip() for arg in args]  # Remove default values and strip whitespace
    return tuple(args)


def verify_positional_args(args: tuple[Any], limit: int):
    """
    Verify a limited number of positional arguments.
    """
    if len(args) > limit:
        raise ValueError(f"A max of {limit} positional arguments are acceptable, not {_argument_names()}.")

    return len(args)


def verify_specified(arg: Any) -> Any:
    """
    Verifies that the argument is not None.
    """
    if arg is None:
        raise ValueError(f"Argument {_argument_names()[0]!r} must be provided, not {arg}.")

    return arg


def verify_not_specified(arg: Any) -> Any:
    """
    Verifies that the argument is None.
    """
    if arg is not None:
        raise ValueError(f"Argument {_argument_names()[0]!r} must not be provided, not {arg}.")

    return arg


def verify_only_one_specified(*args):
    """
    Verifies that only one of the arguments is not None.
    """
    if sum(arg is not None for arg in args) != 1:
        raise ValueError(f"Exactly one of the arguments {_argument_names()} must be provided, not {args}.")

    return args


def verify_at_least_one_specified(*args):
    """
    Verifies that at least one of the arguments is not None.
    """
    if all(arg is None for arg in args):
        raise ValueError(f"At least one of the arguments {_argument_names()} must be provided, not {args}.")

    return args


def verify_isinstance(
    arg: Any,
    types: Any,
    optional: bool = False,
) -> Any:
    """
    Verifies that the argument is an instance of the specified type(s).
    """
    if optional:
        # TODO: Can this be done in a more elegant way?
        try:
            types = list(types)
        except TypeError:
            types = [types]
        types = tuple(types + [type(None)])

    if not isinstance(arg, types):
        raise TypeError(f"Argument {_argument_names()[0]!r} must be an instance of {types}, not {type(arg)}.")

    return arg


def verify_arraylike(
    x: npt.ArrayLike | None,
    dtype: npt.DTypeLike | None = None,
    # Data types
    optional: bool = False,
    int: bool = False,
    float: bool = False,
    complex: bool = False,
    # Value constraints
    real: bool = False,
    imaginary: bool = False,
    negative: bool = False,
    non_negative: bool = False,
    positive: bool = False,
    inclusive_min: float | None = None,
    inclusive_max: float | None = None,
    exclusive_min: float | None = None,
    exclusive_max: float | None = None,
    # Dimension and size constraints
    atleast_1d: bool = False,
    atleast_2d: bool = False,
    atleast_3d: bool = False,
    ndim: int | None = None,
    size: int | None = None,
    size_multiple: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> npt.NDArray:
    """
    Converts the argument to a NumPy array and verifies the conditions.
    """
    if optional and x is None:
        return x

    x = np.asarray(x, dtype=dtype)

    if int:
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int, not {x.dtype}.")
    if float:
        if not (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int or float, not {x.dtype}.")
    if complex:
        if not (
            np.issubdtype(x.dtype, np.integer)
            or np.issubdtype(x.dtype, np.floating)
            or np.issubdtype(x.dtype, np.complexfloating)
        ):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int or float or complex, not {x.dtype}.")

    if real:
        if not np.isrealobj(x):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be real, not complex.")
    if imaginary:
        if not np.iscomplexobj(x):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be complex, not real.")
    if negative:
        if np.any(x >= 0):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be negative, not {x}.")
    if non_negative:
        if np.any(x < 0):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be non-negative, not {x}.")
    if positive:
        if np.any(x <= 0):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be positive, not {x}.")

    if inclusive_min is not None:
        if np.any(x < inclusive_min):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be at least {inclusive_min}, not {x}.")
    if inclusive_max is not None:
        if np.any(x > inclusive_max):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be at most {inclusive_max}, not {x}.")
    if exclusive_min is not None:
        if np.any(x <= exclusive_min):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be greater than {exclusive_min}, not {x}.")
    if exclusive_max is not None:
        if np.any(x >= exclusive_max):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be less than {exclusive_max}, not {x}.")

    if atleast_1d:
        x = np.atleast_1d(x)
    if atleast_2d:
        x = np.atleast_2d(x)
    if atleast_3d:
        x = np.atleast_3d(x)
    if ndim is not None:
        if not x.ndim == ndim:
            raise ValueError(f"Argument {_argument_names()[0]!r} must have {ndim} dimensions, not {x.ndim}.")
    if size is not None:
        if not x.size == size:
            raise ValueError(f"Argument {_argument_names()[0]!r} must have {size} elements, not {x.size}.")
    if size_multiple is not None:
        if not x.size % size_multiple == 0:
            raise ValueError(
                f"Argument {_argument_names()[0]!r} must have a size that is a multiple of {size_multiple}, not {x.size}."
            )
    if shape is not None:
        if not x.shape == shape:
            raise ValueError(f"Argument {_argument_names()[0]!r} must have shape {shape}, not {x.shape}.")

    return x


def verify_scalar(
    x: Any,
    # Data types
    optional: bool = False,
    int: bool = False,
    float: bool = False,
    complex: bool = False,
    # Value constraints
    real: bool = False,
    imaginary: bool = False,
    negative: bool = False,
    non_negative: bool = False,
    positive: bool = False,
    even: bool = False,
    odd: bool = False,
    power_of_two: bool = False,
    inclusive_min: float | None = None,
    inclusive_max: float | None = None,
    exclusive_min: float | None = None,
    exclusive_max: float | None = None,
    # Conversions
    accept_numpy: bool = True,
    convert_numpy: bool = False,
) -> Any:
    """
    Verifies that the argument is a scalar and satisfies the conditions.
    """
    if optional and x is None:
        return x

    if convert_numpy:
        x = convert_to_scalar(x)

    if int:
        if not (isinstance(x, builtins.int) or (accept_numpy and np.issubdtype(x, np.integer))):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int, not {type(x)}.")
    if float:
        if not (isinstance(x, (builtins.int, builtins.float)) or (accept_numpy and np.issubdtype(x, np.floating))):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int or float, not {type(x)}.")
    if complex:
        if not (
            isinstance(x, (builtins.int, builtins.float, builtins.complex))
            or (accept_numpy and np.issubdtype(x, np.complexfloating))
        ):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int or float or complex, not {type(x)}.")

    if real:
        if isinstance(x, builtins.complex):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be real, not complex.")
    if imaginary:
        if not isinstance(x, builtins.complex):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be complex, not real.")

    if negative:
        if x >= 0:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be negative, not {x}.")
    if non_negative:
        if x < 0:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be non-negative, not {x}.")
    if positive:
        if x <= 0:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be positive, not {x}.")
    if even:
        if x % 2 != 0:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be even, not {x}.")
    if odd:
        if x % 2 == 0:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be odd, not {x}.")
    if power_of_two:
        if not (x & (x - 1) == 0):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be a power of two, not {x}.")

    if inclusive_min is not None:
        if x < inclusive_min:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be at least {inclusive_min}, not {x}.")
    if inclusive_max is not None:
        if x > inclusive_max:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be at most {inclusive_max}, not {x}.")
    if exclusive_min is not None:
        if x <= exclusive_min:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be greater than {exclusive_min}, not {x}.")
    if exclusive_max is not None:
        if x >= exclusive_max:
            raise ValueError(f"Argument {_argument_names()[0]!r} must be less than {exclusive_max}, not {x}.")

    return x


def verify_bool(
    x: Any,
    # Conversions
    accept_numpy: bool = True,
    convert_numpy: bool = False,
) -> bool:
    """
    Verifies that the argument is a boolean.
    """
    if convert_numpy:
        x = convert_to_scalar(x)

    if not (isinstance(x, bool) or (accept_numpy and np.issubdtype(x, np.bool_))):
        raise TypeError(f"Argument {_argument_names()[0]!r} must be a bool, not {type(x)}.")

    return x


def verify_literal(
    x: Any,
    literals: Any,
):
    """
    Verifies that the argument is one of the specified literals.
    """
    if not x in literals:
        raise ValueError(f"Argument {_argument_names()[0]!r} must be one of {literals}, not {x!r}.")

    return x


def verify_coprime(
    x: int,
    y: int,
):
    """
    Verifies that the arguments are coprime.
    """
    if np.gcd(x, y) != 1:
        raise ValueError(
            f"Arguments {_argument_names()[0]!r} and {_argument_names()[1]!r} must be coprime, not {x} and {y}."
        )


def verify_condition(
    condition: bool,
):
    """
    Verifies that the condition is satisfied.
    """
    if not condition:
        raise ValueError(f"Arguments must satisfy the condition {_argument_names()[0]!r}.")


def verify_same_shape(
    x: npt.NDArray,
    y: npt.NDArray,
):
    """
    Verifies that the arguments have the same shape.
    """
    if x.shape != y.shape:
        raise ValueError(
            f"Arguments {_argument_names()[0]!r} and {_argument_names()[1]!r} must have the same shape, not {x.shape} and {y.shape}."
        )


def convert_to_scalar(x: Any):
    """
    Converts the input to a scalar if possible.
    """
    if np.isscalar(x) and hasattr(x, "item"):
        x = x.item()

    # TODO: Why is this needed? array(0) with np.int64 does not return true for np.isscalar()
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()

    return x


def convert_output(
    x: Any,
    squeeze: bool = False,
) -> Any:
    """
    Converts the output to a native Python type if scalar.
    """
    if squeeze:
        x = np.squeeze(x)

    x = convert_to_scalar(x)

    return x


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
    module.__all__ = public_members

    return obj


def extend_docstring(method, replace=None, docstring=""):
    """
    A decorator to extend the docstring of `method` with the provided docstring.

    The decorator also finds and replaces and key-value pair in `replace`.
    """
    replace = {} if replace is None else replace

    def decorator(obj):
        parent_docstring = getattr(method, "__doc__", "")
        if parent_docstring is None:
            return obj
        for from_str, to_str in replace.items():
            parent_docstring = parent_docstring.replace(from_str, to_str)
        obj.__doc__ = parent_docstring + "\n" + docstring

        return obj

    return decorator
