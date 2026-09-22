"""Rust expression graphs for a discretised model."""

from __future__ import annotations

import pybamm


def rust_input_parameter_widths(model: pybamm.BaseModel) -> dict[str, int]:
    """Map each input parameter name to its packed width for the Rust backend."""
    return {ip.name: ip._expected_size or 1 for ip in model.input_parameters}


def rust_graph_with_inputs(model: pybamm.BaseModel, input_names):
    """Build an ``ExprGraph`` with ``input_names`` registered in stacking order.

    Registration precedes any lowering so graph input indices match that order.
    """
    from pybamm.rust import ExprGraph

    graph = ExprGraph()
    widths = rust_input_parameter_widths(model)
    for name in input_names:
        graph.input_parameter(name, widths.get(name, 1))
    return graph
