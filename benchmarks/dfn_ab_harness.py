"""Shared DFN A/B harness — CasADi expression tree vs Rust DAG.

Each benchmark goes through `build_dfn_ab(npts)` for one discretised DFN model
and the matched CasADi Function / CompiledModel pair. Both consume the same
`(t, y, p)`, so the A/B is fair: identical math, state size and inputs.

Examples
--------
From a benchmark file in this directory::

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dfn_ab_harness import build_dfn_ab, sample_state

    ab = build_dfn_ab(npts=20)
    y = sample_state(ab)
    p = ab.inputs_array
    res_casadi = ab.casadi_residual(0.0, y, p)
    res_rust = ab.rust_model.rhs(0.0, y, p)
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass

import numpy as np

import pybamm


@dataclass
class DFNAB:
    """Container for the matched CasADi/Rust DFN evaluation pair.

    All callables operate on the same `(t, y, p)` tuple, where `p` is a
    1-D numpy array of input-parameter values in declaration order.
    """

    n_states: int
    n_inputs: int
    npts: int
    y0: np.ndarray
    inputs_dict: dict[str, float]
    inputs_array: np.ndarray
    # CasADi side — `casadi.Function` instances bound to the discretised model.
    casadi_residual: object
    casadi_jacobian: object
    # Rust side — single CompiledModel exposing all needed evals.
    rust_model: object
    # Sparsity, useful for assembled-Jacobian benchmarks.
    csc_colptrs: np.ndarray
    csc_rowinds: np.ndarray
    nnz: int


def _make_var_pts(model: pybamm.BaseModel, npts: int) -> dict:
    """Override scalar var_pts with `npts`; preserve direction grids (y, z)."""
    out = {}
    for k, v in model.default_var_pts.items():
        if isinstance(v, numbers.Number) and v > 1 and k not in {"y", "z"}:
            out[k] = npts
        else:
            out[k] = v
    return out


def build_dfn_ab(
    npts: int = 20,
    *,
    with_input_current: bool = True,
    parameter_set: str = "Chen2020",
) -> DFNAB:
    """Build a discretised DFN and matched CasADi/Rust evaluation pair.

    Parameters
    ----------
    npts
        Number of finite-volume points in each spatial direction (other than
        the unused y/z directions). 5 -> 92 states, 10 -> 282, 20 -> 962
        (default), 30 -> 2042. Used to drive scale sweeps.
    with_input_current
        If True, parameterise `Current function [A]` as InputParameter "I" so
        the benchmark exercises the input-vector path. Defaults to True.
    parameter_set
        Name of the PyBaMM parameter set. Defaults to "Chen2020".
    """
    from pybamm.rust import CompiledModel, ExprGraph

    model = pybamm.lithium_ion.DFN()
    model.events = []  # Rust path doesn't support root-finding events yet.

    param = pybamm.ParameterValues(parameter_set)
    if with_input_current:
        param["Current function [A]"] = pybamm.InputParameter("I")
        inputs_dict = {"I": 0.5}
    else:
        inputs_dict = {}

    var_pts = _make_var_pts(model, npts)

    model.convert_to_format = "casadi"
    sim = pybamm.Simulation(
        model,
        parameter_values=param,
        var_pts=var_pts,
        solver=pybamm.IDAKLUSolver(),
    )
    sim.build()
    built = sim.built_model

    # Populate `rhs_algebraic_eval`, `jac_rhs_algebraic_eval`, etc. on the model.
    sim.solver.set_up(built, inputs=inputs_dict)

    n_states = built.len_rhs_and_alg
    n_inputs = len(inputs_dict)
    inputs_array = np.array(
        [float(inputs_dict[k]) for k in inputs_dict], dtype=np.float64
    )
    y0 = np.asarray(built.y0.full() if hasattr(built.y0, "full") else built.y0).reshape(
        -1
    )

    # Convert the same Symbol `_set_up_rust` would and build a CompiledModel
    # directly, without going through the solver-group wrapper.
    if built.len_alg > 0:
        full_sym = pybamm.numpy_concatenation(
            built.concatenated_rhs, built.concatenated_algebraic
        )
    else:
        full_sym = built.concatenated_rhs
    graph = ExprGraph()
    rhs_expr = full_sym.to_rust(graph, {})

    mass = built.mass_matrix.entries  # scipy CSR
    rust_model = CompiledModel.from_expr(
        graph,
        rhs_expr,
        mass.data.astype(np.float64),
        mass.indptr.astype(np.int64),
        mass.indices.astype(np.int64),
        n_inputs=n_inputs,
    )

    csc_colptrs, csc_rowinds = rust_model.csc_sparsity_pattern()

    return DFNAB(
        n_states=n_states,
        n_inputs=n_inputs,
        npts=npts,
        y0=y0,
        inputs_dict=inputs_dict,
        inputs_array=inputs_array,
        casadi_residual=built.rhs_algebraic_eval,
        casadi_jacobian=built.jac_rhs_algebraic_eval,
        rust_model=rust_model,
        csc_colptrs=np.asarray(csc_colptrs, dtype=np.int64),
        csc_rowinds=np.asarray(csc_rowinds, dtype=np.int64),
        nnz=int(rust_model.nnz),
    )


def sample_state(
    ab: DFNAB, *, perturb: float = 0.0, seed: int | None = None
) -> np.ndarray:
    """Return `ab.y0` optionally perturbed by `perturb * randn(n_states)`."""
    if perturb <= 0:
        return ab.y0.copy()
    rng = np.random.default_rng(seed)
    return ab.y0 + perturb * rng.standard_normal(ab.n_states)


def casadi_jacobian_dense(ab: DFNAB, t: float, y: np.ndarray) -> np.ndarray:
    """Convert CasADi sparse Jacobian to a dense numpy array (for parity)."""
    J = ab.casadi_jacobian(t, y, ab.inputs_array)
    return np.asarray(J)
