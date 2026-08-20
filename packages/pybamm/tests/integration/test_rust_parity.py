# tests/integration/test_rust_parity.py
"""Dual-backend parity tests: verify Rust evaluation matches CasADi."""

import numpy as np

import pybamm

# Import fixtures from conftest_rust
pytest_plugins = ["tests.conftest_rust"]


class TestRustCasadiParity:
    """Tests that verify Rust backend produces same results as CasADi."""

    def test_scalar_arithmetic(self, dual_backend_compare):
        """Basic scalar arithmetic."""
        a = pybamm.Scalar(2.0)
        b = pybamm.Scalar(3.0)

        dual_backend_compare(a + b)
        dual_backend_compare(a - b)
        dual_backend_compare(a * b)
        dual_backend_compare(a / b)
        dual_backend_compare(a**b)

    def test_unary_functions(self, dual_backend_compare):
        """Unary math functions."""
        x = pybamm.InputParameter("x")

        dual_backend_compare(pybamm.sqrt(x), inputs={"x": 4.0})
        dual_backend_compare(pybamm.exp(x), inputs={"x": 1.0})
        dual_backend_compare(pybamm.log(x), inputs={"x": np.e})
        dual_backend_compare(pybamm.sin(x), inputs={"x": np.pi / 6})
        dual_backend_compare(pybamm.cos(x), inputs={"x": np.pi / 3})
        dual_backend_compare(pybamm.tanh(x), inputs={"x": 1.0})

    def test_state_vector_operations(self, dual_backend_compare):
        """StateVector arithmetic."""
        sv = pybamm.StateVector(slice(0, 3))
        expr = pybamm.Scalar(2.0) * sv + pybamm.Scalar(1.0)

        y = np.array([1.0, 2.0, 3.0])
        dual_backend_compare(expr, y=y)

    def test_interpolation_1d(self, dual_backend_compare):
        """1D linear interpolation."""
        x_data = np.linspace(0, 1, 50)
        y_data = 2 * x_data
        sv = pybamm.StateVector(slice(0, 2))
        interp = pybamm.Interpolant(x_data, y_data, sv)

        y = np.array([0.25, 0.75])
        dual_backend_compare(interp, y=y)

    def test_interpolation_1d_vector_valued_y(self, dual_backend_compare):
        """1D interpolation with vector-valued y (one output per column)."""
        x_data = np.linspace(0, 1, 50)
        y_data = np.column_stack([2 * x_data, np.sin(x_data)])
        sv = pybamm.StateVector(slice(0, 1))

        y = np.array([0.25])
        for interpolator in ["linear", "cubic", "pchip"]:
            interp = pybamm.Interpolant(x_data, y_data, sv, interpolator=interpolator)
            dual_backend_compare(interp, y=y)

    def test_concatenation(self, dual_backend_compare):
        """Vector concatenation."""
        v1 = pybamm.Vector(np.array([1.0, 2.0]))
        v2 = pybamm.Vector(np.array([3.0, 4.0, 5.0]))
        expr = pybamm.NumpyConcatenation(v1, v2)

        dual_backend_compare(expr)

    def test_sparse_matmul(self, dual_backend_compare):
        """Sparse matrix-vector multiplication."""
        from scipy.sparse import csr_matrix

        data = np.array([1.0, 2.0, 3.0])
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        sparse_mat = csr_matrix((data, (row, col)), shape=(3, 3))

        mat = pybamm.Matrix(sparse_mat)
        sv = pybamm.StateVector(slice(0, 3))
        expr = mat @ sv

        y = np.array([1.0, 2.0, 3.0])
        dual_backend_compare(expr, y=y)

    def test_composed_expression(self, dual_backend_compare):
        """Complex composed expression."""
        sv = pybamm.StateVector(slice(0, 1))
        t = pybamm.Time()
        expr = pybamm.sqrt(sv**2 + pybamm.Scalar(1.0)) * pybamm.exp(-t)

        y = np.array([3.0])
        dual_backend_compare(expr, t=2.0, y=y, rtol=1e-8)

    def test_conditional_parity(self, dual_backend_compare):
        """Conditional branch selection matches CasADi."""
        selector = pybamm.InputParameter("s")
        branch1 = pybamm.InputParameter("a") * pybamm.Scalar(2.0)
        branch2 = pybamm.InputParameter("b") + pybamm.Scalar(5.0)
        branch3 = pybamm.Scalar(42.0)
        expr = pybamm.Conditional(selector, branch1, branch2, branch3)

        # Test each branch
        dual_backend_compare(
            expr, inputs={"s": 1.0, "a": 10.0, "b": 20.0}
        )  # branch1: 20
        dual_backend_compare(
            expr, inputs={"s": 2.0, "a": 10.0, "b": 20.0}
        )  # branch2: 25
        dual_backend_compare(
            expr, inputs={"s": 3.0, "a": 10.0, "b": 20.0}
        )  # branch3: 42
        dual_backend_compare(
            expr, inputs={"s": 0.0, "a": 10.0, "b": 20.0}
        )  # no branch: 0
