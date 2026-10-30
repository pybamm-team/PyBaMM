#
# Tests for the block preconditioner backend
#
import numpy as np
import pytest
from pybammsolvers import idaklu
from scipy.sparse import csc_matrix

import pybamm
from tests.unit.test_solvers.test_preconditioners.test_tree import (
    C_E,
    C_N,
    C_P,
    PHI_E,
    ExampleJacobi,
    build_dfn,
)

P = pybamm.preconditioners
LAST_GAUSS_SEIDEL = P.Stage(None, composition="multiplicative")


def iteration_matrix(pattern, seed=0):
    """Random data on the pattern, in its CSC order, with a dominant diagonal."""
    rng = np.random.default_rng(seed)
    data = rng.random(pattern.nnz) - 0.5
    rows = pattern.indices
    cols = np.repeat(np.arange(pattern.shape[1]), np.diff(pattern.indptr))
    data[rows == cols] += 10.0
    return data, csc_matrix(
        (data, pattern.indices, pattern.indptr), shape=pattern.shape
    )


def apply(preconditioner, seed=0):
    model, pattern = build_dfn(preconditioner)
    analysis = P.analyse(model.preconditioner, pattern)
    data, matrix = iteration_matrix(pattern, seed)
    preconditioner = idaklu.BlockPreconditioner(analysis)
    assert preconditioner.setup(data) == 0
    r = np.random.default_rng(seed + 1).random(matrix.shape[0])
    z = preconditioner.solve(r)
    return analysis, z, np.linalg.norm(matrix @ z - r) / np.linalg.norm(r)


class TestBlockBackend:
    @pytest.mark.parametrize(
        "preconditioner",
        [
            {},
            {C_N: P.Tridiagonal(stage=0), C_P: P.Tridiagonal(stage=0)},
            {C_N: P.Tridiagonal(stage=0), C_P: P.KLU(stage=1)},
            {C_N: P.KLU(stage=0), C_P: P.Tridiagonal(stage=0)},
        ],
    )
    def test_exact_trees_solve_exactly(self, preconditioner):
        analysis, _, residual = apply(preconditioner)
        assert analysis.exact
        assert residual < 1e-12

    @pytest.mark.parametrize(
        "preconditioner",
        [
            {PHI_E: P.ILU()},
            {PHI_E: P.ILU(fill=2)},
            {PHI_E: P.Jacobi()},
            {
                PHI_E: P.ILU(stage=LAST_GAUSS_SEIDEL),
                C_E: P.KLU(stage=LAST_GAUSS_SEIDEL),
            },
            {C_N: P.Tridiagonal(stage=0), PHI_E: P.Jacobi(stage=0)},
            {C_N: P.Tridiagonal(stage=P.Stage(0, schur_complement="diagonal"))},
            {C_N: P.Tridiagonal(stage=P.Stage(0, schur_complement="none"))},
        ],
    )
    def test_approximate_trees(self, preconditioner):
        analysis, _, residual = apply(preconditioner)
        assert not analysis.exact
        assert np.isfinite(residual)
        assert residual < 1

    def test_external_block_solver(self):
        # The example plugin is a Jacobi solve, so it matches the built-in one
        _, plugin, _ = apply({PHI_E: ExampleJacobi()})
        _, builtin, _ = apply({PHI_E: P.Jacobi()})
        np.testing.assert_allclose(plugin, builtin, rtol=1e-14)

    def test_singular_block_fails_setup(self):
        model, pattern = build_dfn({PHI_E: P.Jacobi()})
        data, _ = iteration_matrix(pattern)
        data[:] = 0.0
        preconditioner = idaklu.BlockPreconditioner(
            P.analyse(model.preconditioner, pattern)
        )
        assert preconditioner.setup(data) == 1

    def test_analysis_errors(self):
        _, pattern = build_dfn()
        indptr = pattern.indptr.astype(np.int64)
        indices = pattern.indices.astype(np.int64)
        spec = {"type": "leaf", "solver": "klu", "indices": np.arange(3)}
        with pytest.raises(ValueError, match=r"cover every state exactly once"):
            idaklu.BlockAnalysis(spec, indptr, indices)
        everything = np.arange(pattern.shape[0])
        spec = {"type": "leaf", "solver": "external", "indices": everything}
        with pytest.raises(TypeError):
            idaklu.BlockAnalysis(
                {**spec, "capsule": 1, "options": "{}"}, indptr, indices
            )

    def test_read_entries(self):
        model, pattern = build_dfn()
        # One exact KLU block reads every entry
        everything = P.Leaf(np.arange(pattern.shape[0]), P.KLU(), ("all",))
        np.testing.assert_array_equal(
            P.analyse(everything, pattern).read_entries, np.arange(pattern.nnz)
        )
        # Additive blocks read only their own diagonal blocks
        tree = P.compile_tree(model, {C_N: P.ILU()})
        read = P.analyse(tree, pattern).read_entries
        block = np.zeros(pattern.shape[0], dtype=np.int64)
        for k, child in enumerate(tree.children):
            block[child.indices] = k
        coo = csc_matrix(pattern).tocoo()
        same_block = block[coo.row] == block[coo.col]
        np.testing.assert_array_equal(read, np.flatnonzero(same_block))

    def test_analyse_jacobian(self):
        jacobian = csc_matrix(np.array([[0, 1, 1], [1, 1, 0], [0, 1, 1]]))
        # The mass matrix adds the first state's diagonal entry
        mass_matrix = csc_matrix(np.diag([1.0, 0.0, 0.0]))
        tree = P.Additive(
            (
                P.Leaf(np.array([0, 1]), P.KLU(), ("a",)),
                P.Leaf(np.array([2]), P.KLU(), ("b",)),
            )
        )
        result = P.analyse_jacobian(tree, jacobian, mass_matrix)
        assert result.pattern[0, 0]
        assert result.pattern.nnz == jacobian.nnz + 1
        # Within-block entries of J, in its CSC order: (1,0), (0,1), (1,1), (2,2)
        np.testing.assert_array_equal(result.entries, [0, 1, 2, 5])
