"""DFN-driven Rust IDAKLU DAE parity tests."""

import numpy as np
import pytest

from tests.integration.dfn_ab_harness import (
    build_dfn_ab,
    casadi_jacobian_dense,
    sample_state,
)

pytest.importorskip("casadi")


@pytest.fixture(scope="module")
def ab_small():
    """Small DFN (npts=5), still mixed RHS/algebraic."""
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
    assert ab_small.n_states > 50, "DFN should yield > 50 states even at npts=5"
    ids = np.asarray(ab_small.rust_model.algebraic_ids())
    jac_stats = ab_small.rust_model.jacobian_stats()
    assert ids.shape == (ab_small.n_states,)
    assert np.any(ids == 1.0), "Expected at least one differential row"
    assert np.any(ids == 0.0), "Expected at least one algebraic row"
    assert jac_stats["n_colors"] > 0
    assert jac_stats["nnz"] == ab_small.nnz


def test_dfn_residual_parity(ab_small):
    rng = np.random.default_rng(7)
    for _ in range(3):
        y = sample_state(ab_small, perturb=1e-3, seed=rng.integers(0, 1 << 32))
        rc = np.asarray(
            ab_small.casadi_residual(0.0, y, ab_small.inputs_array)
        ).reshape(-1)
        rr = ab_small.rust_model.rhs(0.0, y, ab_small.inputs_array)
        np.testing.assert_allclose(rr, rc, rtol=1e-10, atol=1e-12)


def test_dfn_jacobian_alg_rows_parity(ab_small):
    y = sample_state(ab_small)
    Jc = casadi_jacobian_dense(ab_small, 0.0, y)
    Jr = _assemble_rust_jacobian_dense(ab_small, 0.0, y)
    ids = np.asarray(ab_small.rust_model.algebraic_ids())
    alg_mask = ids == 0.0
    np.testing.assert_allclose(Jr[alg_mask], Jc[alg_mask], rtol=1e-10, atol=1e-12)


def test_dfn_jacobian_rhs_rows_parity(ab_small):
    y = sample_state(ab_small)
    Jc = casadi_jacobian_dense(ab_small, 0.0, y)
    Jr = _assemble_rust_jacobian_dense(ab_small, 0.0, y)
    ids = np.asarray(ab_small.rust_model.algebraic_ids())
    rhs_mask = ids == 1.0
    np.testing.assert_allclose(Jr[rhs_mask], Jc[rhs_mask], rtol=1e-10, atol=1e-12)


def test_dfn_jacobian_nonzero_cj_merges_mass(ab_small):
    """DFN mass is diag(algebraic_ids), so J(cj) must equal J(0) - cj * M."""
    y = sample_state(ab_small)
    cj = 1.5
    J0 = _assemble_rust_jacobian_dense(ab_small, 0.0, y)
    Jcj = _assemble_rust_jacobian_dense(ab_small, 0.0, y, cj=cj)

    ids = np.asarray(ab_small.rust_model.algebraic_ids())
    assert np.any(ids == 1.0) and np.any(ids == 0.0)
    M = np.diag(ids)
    np.testing.assert_allclose(Jcj, J0 - cj * M, rtol=1e-12, atol=1e-12)
