"""Rust expression graphs for a discretised model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pybamm

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pybamm.rust import CompiledFunction, ExprGraph


def rust_input_parameter_widths(model: pybamm.BaseModel) -> dict[str, int]:
    """Map each input parameter name to its packed width for the Rust backend."""
    return {ip.name: ip._expected_size or 1 for ip in model.input_parameters}


def reject_y_dot(function: CompiledFunction) -> None:
    """Refuse a compiled function that reads the state derivatives, ``y_dot``.

    Raises
    ------
    :class:`pybamm.ModelError`
        If ``function`` reads ``y_dot``, as the CasADi conversion does without it.
    """
    if function.uses_y_dot:
        raise pybamm.ModelError(
            "Must provide a 'y_dot' for converting state vectors differentiated "
            "in time, e.g. by `symbol.diff(pybamm.t)`"
        )


def rust_graph_with_inputs(
    model: pybamm.BaseModel, input_names: Iterable[str]
) -> ExprGraph:
    """Build an ``ExprGraph`` with ``input_names`` registered in stacking order.

    Registration precedes any lowering so graph input indices match that order.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The discretised model whose input parameter widths are registered.
    input_names : iterable of str
        Input parameter names, in the order the inputs are stacked.

    Returns
    -------
    :class:`pybamm.rust.ExprGraph`
        A graph with each input registered at its packed width.
    """
    from pybamm.rust import ExprGraph

    graph = ExprGraph()
    widths = rust_input_parameter_widths(model)
    for name in input_names:
        graph.input_parameter(name, widths.get(name, 1))
    return graph
