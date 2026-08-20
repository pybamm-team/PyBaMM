"""DFN-driven Rust IDAKLU DAE parity tests.

DFN is a real DAE — 962 differential states + 100 algebraic constraints at
default `var_pts` — so each parity assertion exercises the full
ABI/converter/algebraic-IC path on production-shape math.
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("casadi")


# Reuse the benchmarks/ DFN A/B harness — same builder used by the timing
# benchmarks, so test parity and benchmark fairness stay aligned.
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "benchmarks")
    ),
)
from dfn_ab_harness import (
    build_dfn_ab,
    casadi_jacobian_dense,
    sample_state,
)


@pytest.fixture(scope="module")
def ab_small():
    """Small DFN (npts=5 -> 92 states) — fast, still mixed RHS/algebraic."""
    return build_dfn_ab(npts=5)


def _assemble_rust_jacobian_dense(ab, t, y, cj=0.0):
    """Run Rust assembled Jacobian and reconstruct dense (n,n) numpy matrix."""
    jac_data = np.zeros(ab.nnz)
    ab.rust_model.assemble_jacobian_csc_into(t, y, cj, ab.inputs_array, jac_data)
    Jr = np.zeros((ab.n_states, ab.n_states))
    for col in range(ab.n_states):
        for k in range(ab.csc_colptrs[col], ab.csc_colptrs[col + 1]):
            Jr[ab.csc_rowinds[k], col] = jac_data[k]
    return Jr


def test_dfn_set_up_succeeds(ab_small):
    """Rust setup must produce a model with consistent metadata for DFN."""
    assert ab_small.n_states > 50, "DFN should yield > 50 states even at npts=5"
    ids = np.asarray(ab_small.rust_model.algebraic_ids())
    jac_stats = ab_small.rust_model.jacobian_stats()
    assert ids.shape == (ab_small.n_states,)
    # Mixed DAE: must have both rhs (1.0) and algebraic (0.0) ids.
    assert np.any(ids == 1.0), "Expected at least one differential row"
    assert np.any(ids == 0.0), "Expected at least one algebraic row"
    assert jac_stats["n_colors"] > 0
    assert jac_stats["nnz"] == ab_small.nnz
    assert jac_stats["strategy"] == "coloring"


def test_dfn_residual_parity(ab_small):
    """Rust rhs and CasADi rhs_algebraic_eval agree on f(t, y, p).

    Both backends consume the *same* discretised model (built once by the
    harness), so residual parity verifies the Rust expression converter
    produced a graph numerically equivalent to CasADi's symbolic form.
    """
    rng = np.random.default_rng(7)
    for _ in range(3):
        y = sample_state(ab_small, perturb=1e-3, seed=rng.integers(0, 1 << 32))
        rc = np.asarray(
            ab_small.casadi_residual(0.0, y, ab_small.inputs_array)
        ).reshape(-1)
        rr = ab_small.rust_model.rhs(0.0, y, ab_small.inputs_array)
        np.testing.assert_allclose(rr, rc, rtol=1e-10, atol=1e-12)


def test_dfn_jacobian_alg_rows_parity(ab_small):
    """Rust assembled Jacobian matches CasADi exactly on algebraic rows."""
    y = sample_state(ab_small)
    Jc = casadi_jacobian_dense(ab_small, 0.0, y)
    Jr = _assemble_rust_jacobian_dense(ab_small, 0.0, y)
    ids = np.asarray(ab_small.rust_model.algebraic_ids())
    alg_mask = ids == 0.0
    np.testing.assert_allclose(Jr[alg_mask], Jc[alg_mask], rtol=1e-10, atol=1e-12)


def test_dfn_jacobian_rhs_rows_parity(ab_small):
    """Rust assembled Jacobian matches CasADi exactly on RHS rows."""
    y = sample_state(ab_small)
    Jc = casadi_jacobian_dense(ab_small, 0.0, y)
    Jr = _assemble_rust_jacobian_dense(ab_small, 0.0, y)
    ids = np.asarray(ab_small.rust_model.algebraic_ids())
    rhs_mask = ids == 1.0
    np.testing.assert_allclose(Jr[rhs_mask], Jc[rhs_mask], rtol=1e-10, atol=1e-12)


def test_dfn_jacobian_nonzero_cj_merges_mass(ab_small):
    """Rust-side cj*mass merge with a non-identity (DAE) mass matrix.

    DFN mass is diag(algebraic_ids) — ones on differential rows, zeros on
    algebraic rows — so nonzero cj must shift exactly the differential
    diagonal: J(cj) == J(0) - cj * M. This exercises the Rust merged-mass
    scatter (mass_to_csc_map) itself, not a scipy reconstruction.
    """
    y = sample_state(ab_small)
    cj = 1.5
    J0 = _assemble_rust_jacobian_dense(ab_small, 0.0, y)
    Jcj = _assemble_rust_jacobian_dense(ab_small, 0.0, y, cj=cj)

    ids = np.asarray(ab_small.rust_model.algebraic_ids())
    # mixed DAE: the merge must bite on differential rows and skip algebraic
    assert np.any(ids == 1.0) and np.any(ids == 0.0)
    M = np.diag(ids)
    np.testing.assert_allclose(Jcj, J0 - cj * M, rtol=1e-12, atol=1e-12)
