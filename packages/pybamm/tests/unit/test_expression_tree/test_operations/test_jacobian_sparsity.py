#
# Tests for the structural Jacobian sparsity and the CasADi Jacobian assembled from it
#
import casadi
import numpy as np
import pytest
import scipy.sparse

import pybamm


def sparsity(symbol, n_states):
    return pybamm.JacobianSparsity(n_states).pattern(symbol)


def pattern(symbol, n_states):
    return sparsity(symbol, n_states).toarray()


def dense_row_and_column(n):
    """A field with a stencil, a mean over the field and a scalar every row reads."""
    field = pybamm.StateVector(slice(0, n))
    scalar = pybamm.StateVector(slice(n, n + 1))
    stencil = pybamm.Matrix(
        scipy.sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n), format="csr")
    )
    mean = pybamm.Matrix(scipy.sparse.csr_matrix(np.full((1, n), 1 / n)))
    return pybamm.NumpyConcatenation(
        stencil @ field + scalar * pybamm.exp(field), scalar - mean @ field
    )


def casadi_pair(symbol, n_states):
    y = casadi.MX.sym("y", n_states)
    expression = symbol.to_casadi(casadi.MX.sym("t"), y)
    return expression, y


def assert_jacobians_equal(symbol, n_states, y0):
    expression, y = casadi_pair(symbol, n_states)
    assembled = pybamm.casadi_jacobian(expression, y, sparsity(symbol, n_states))
    reference = casadi.jacobian(expression, y)
    values = casadi.Function("j", [y], [assembled, reference])(y0)
    np.testing.assert_allclose(
        values[0].full(), values[1].full(), rtol=1e-12, atol=1e-12
    )


class TestJacobianSparsity:
    def test_state_vectors_and_constants(self):
        u = pybamm.StateVector(slice(0, 2), slice(3, 4))
        np.testing.assert_array_equal(
            pattern(u, 4), [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )
        np.testing.assert_array_equal(pattern(pybamm.Vector(np.ones(2)), 4), 0)
        np.testing.assert_array_equal(pattern(pybamm.t * pybamm.Scalar(2), 4), 0)

    def test_matrix_product_ignores_stored_zeros(self):
        matrix = scipy.sparse.csr_matrix(
            (np.array([1.0, 0.0]), (np.array([0, 1]), np.array([0, 1]))), shape=(2, 2)
        )
        y = pybamm.StateVector(slice(0, 2))
        np.testing.assert_array_equal(
            pattern(pybamm.Matrix(matrix) @ y, 2), [[1, 0], [0, 0]]
        )
        with pytest.raises(NotImplementedError, match="constant left factor"):
            sparsity(
                pybamm.StateVector(slice(0, 1)) @ pybamm.StateVector(slice(1, 2)), 2
            )

    def test_elementwise_broadcast_and_reduction(self):
        u = pybamm.StateVector(slice(0, 2))
        s = pybamm.StateVector(slice(2, 3))
        np.testing.assert_array_equal(
            pattern(pybamm.exp(u) * s, 3), [[1, 0, 1], [0, 1, 1]]
        )
        np.testing.assert_array_equal(pattern(pybamm.max(u), 3), [[1, 1, 0]])
        np.testing.assert_array_equal(pattern(pybamm.sign(u) + 1, 3), 0)
        np.testing.assert_array_equal(pattern((u > s) * u, 3), [[1, 0, 0], [0, 1, 0]])

    def test_index_and_concatenation(self):
        y = pybamm.StateVector(slice(0, 3))
        np.testing.assert_array_equal(
            pattern(pybamm.Index(y, slice(1, 3)), 3), [[0, 1, 0], [0, 0, 1]]
        )
        stacked = pybamm.NumpyConcatenation(
            pybamm.StateVector(slice(2, 3)), pybamm.StateVector(slice(0, 1))
        )
        np.testing.assert_array_equal(pattern(stacked, 3), [[0, 0, 1], [1, 0, 0]])

    def test_conditional_uses_branches(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.StateVector(slice(1, 2))
        np.testing.assert_array_equal(
            pattern(pybamm.Conditional(pybamm.t, a, 2 * b), 3), [[1, 1, 0]]
        )

    def test_unsupported_node(self):
        y = pybamm.StateVector(slice(0, 2))
        with pytest.raises(NotImplementedError, match="No Jacobian sparsity rule"):
            sparsity(pybamm.Transpose(y), 2)

    @pytest.mark.parametrize(
        "model, options",
        [
            (pybamm.lithium_ion.DFN, {}),
            (
                pybamm.lithium_ion.SPMe,
                {"current collector": "potential pair", "dimensionality": 1},
            ),
        ],
    )
    def test_battery_models_match_casadi(self, model, options):
        sim = pybamm.Simulation(model(options))
        sim.build()
        built = sim.built_model
        n = built.len_rhs_and_alg
        symbol = pybamm.NumpyConcatenation(
            built.concatenated_rhs, built.concatenated_algebraic
        )
        expression, y = casadi_pair(symbol, n)
        y0 = (
            built.concatenated_initial_conditions.evaluate().ravel()
            * np.random.default_rng(0).uniform(0.9, 1.1, n)
        )
        jacobian = casadi.Function("j", [y], [casadi.jacobian(expression, y)])(y0)
        numerical = scipy.sparse.csc_matrix(jacobian.sparse())
        numerical.eliminate_zeros()
        structural = sparsity(symbol, n)
        # Every nonzero is in the pattern, and the pattern is no wider than CasADi's
        assert (numerical.astype(bool) > structural).nnz == 0
        detected = casadi.jacobian_sparsity(expression, y)
        casadi_pattern = scipy.sparse.csc_matrix(
            (np.ones(detected.nnz(), dtype=bool), detected.get_triplet()), shape=(n, n)
        )
        assert (structural > casadi_pattern).nnz == 0
        assert_jacobians_equal(symbol, n, y0)


class TestCasadiJacobian:
    def test_dense_row_and_column(self):
        n = 50
        symbol = dense_row_and_column(n)
        assert_jacobians_equal(symbol, n + 1, np.linspace(0.5, 1.5, n + 1))

    def test_few_sweeps_with_dense_row_and_column(self):
        n = 400
        symbol = dense_row_and_column(n)
        expression, y = casadi_pair(symbol, n + 1)
        jacobian = pybamm.casadi_jacobian(expression, y, sparsity(symbol, n + 1))
        # One-directional colouring needs n + 1 sweeps here; this needs a handful
        assembled = casadi.Function("j", [y], [jacobian])
        reference = casadi.Function("j", [y], [casadi.jacobian(expression, y)])
        assert assembled.n_instructions() < reference.n_instructions()

    def test_empty_pattern(self):
        y = casadi.MX.sym("y", 2)
        jacobian = pybamm.casadi_jacobian(
            casadi.MX.ones(3), y, scipy.sparse.csr_matrix((3, 2), dtype=bool)
        )
        assert jacobian.shape == (3, 2)
        assert jacobian.nnz() == 0

    def test_model_uses_structural_jacobian(self):
        n = 20
        symbol = dense_row_and_column(n)
        expression, y = casadi_pair(symbol, n + 1)
        model = pybamm.BaseModel()
        jacobian = model.build_casadi_jacobian(expression, y, sparsity(symbol, n + 1))
        assert jacobian.nnz() == sparsity(symbol, n + 1).nnz
        plain = model.build_casadi_jacobian(expression, y)
        assert plain.sparsity() == casadi.jacobian(expression, y).sparsity()
