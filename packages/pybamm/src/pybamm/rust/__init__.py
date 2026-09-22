"""Access point for the compiled Rust extension; import from here, not ``_core``."""

import contextlib
import os
import sys


@contextlib.contextmanager
def _global_symbol_visibility():
    """Load extensions ``RTLD_GLOBAL`` on Linux so pybammsolvers can ``dlsym`` the Rust FFI."""
    if not sys.platform.startswith("linux"):
        yield
        return

    previous_flags = sys.getdlopenflags()
    sys.setdlopenflags(previous_flags | os.RTLD_GLOBAL)
    try:
        yield
    finally:
        sys.setdlopenflags(previous_flags)


with _global_symbol_visibility():
    from pybamm.rust._core import (
        CompiledFunction,
        CompiledJacobian,
        Expr,
        ExprGraph,
    )

__all__ = [
    "CompiledFunction",
    "CompiledJacobian",
    "Expr",
    "ExprGraph",
]
