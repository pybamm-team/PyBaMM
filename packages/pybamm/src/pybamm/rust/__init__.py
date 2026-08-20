"""Access point for the compiled Rust extension.

``pybamm.rust._core`` is the built extension module; import its classes from
``pybamm.rust`` rather than reaching into ``_core``, so the private module stays
free to move. Importing this package fails if the extension was not built, which
is why callers that must tolerate its absence import it inside a function.
"""

import contextlib
import os
import sys


@contextlib.contextmanager
def _global_symbol_visibility():
    """Open extension modules into the process-global symbol scope on Linux.

    pybammsolvers cannot link against the Rust core (the dependency only flows
    the other way), so it resolves the FFI entry points with
    ``dlsym(RTLD_DEFAULT)``. That only finds symbols from libraries loaded with
    ``RTLD_GLOBAL``, and CPython defaults to ``RTLD_LOCAL`` on Linux. macOS
    exports them globally already, so this is a no-op there.
    """
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
        CompiledFunctionGroup,
        CompiledJacobian,
        CompiledModel,
        EvaluatorPool,
        Expr,
        ExprGraph,
        PreparedSolver,
        SolveOutcome,
        SolverStatistics,
        default_solver_options,
    )

# Without this, ``contextlib``/``os``/``sys`` are also public names here.
__all__ = [
    "CompiledFunction",
    "CompiledFunctionGroup",
    "CompiledJacobian",
    "CompiledModel",
    "EvaluatorPool",
    "Expr",
    "ExprGraph",
    "PreparedSolver",
    "SolveOutcome",
    "SolverStatistics",
    "default_solver_options",
]
