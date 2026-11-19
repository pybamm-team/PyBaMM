#
# Compile expressions into functions of named inputs that return their values and
# Jacobian
#
from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

import casadi
import numpy as np
import numpy.typing as npt

import pybamm


def compile_input_jacobian(
    symbols: Sequence[pybamm.Symbol],
    input_names: Sequence[str],
    convert_to_format: str | None = None,
) -> Callable[
    [npt.NDArray[np.float64]],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
]:
    """
    Compile scalar symbols into a function of named scalar inputs that returns their
    values and their Jacobian with respect to those inputs.

    Parameters
    ----------
    symbols : sequence of :class:`pybamm.Symbol`
        Scalar expressions whose only leaves are scalars and the input parameters
        named in ``input_names``.
    input_names : sequence of str
        The names of the scalar inputs, in the order of the Jacobian's columns.
    convert_to_format : str, optional
        The backend to compile with. Default is the default
        ``convert_to_format`` of :class:`pybamm.BaseModel`. Formats without their
        own implementation use CasADi.

    Returns
    -------
    callable
        A function of the input values, ordered as ``input_names``, returning the
        values of the symbols, with shape ``(len(symbols),)``, and their Jacobian,
        with shape ``(len(symbols), len(input_names))``.
    """
    if convert_to_format is None:
        convert_to_format = pybamm.BaseModel._DEFAULT_CONVERT_TO_FORMAT
    return _compile(tuple(symbols), tuple(input_names), convert_to_format)


@lru_cache(maxsize=16)
def _compile(symbols, input_names, convert_to_format):
    # Every format compiles with CasADi until it has its own implementation
    return _compile_casadi(symbols, input_names)


def _compile_casadi(symbols, input_names):
    p = casadi.MX.sym("p", len(input_names))
    inputs = {name: p[k] for k, name in enumerate(input_names)}
    cache = {}
    value = casadi.vertcat(
        *[s.to_casadi(inputs=inputs, casadi_symbols=cache) for s in symbols]
    )
    function = casadi.Function(
        "input_jacobian", [p], [value, casadi.jacobian(value, p)]
    )

    def evaluate(values):
        value, jacobian = function(values)
        return value.full().ravel(), jacobian.full()

    return evaluate
