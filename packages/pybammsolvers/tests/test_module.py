"""Unit tests for pybammsolvers core functionality."""

from __future__ import annotations

import pytest


class TestPackageStructure:
    """Test pybammsolvers package structure and metadata."""

    pytestmark = pytest.mark.unit

    def test_package_import(self):
        """
        Verify pybammsolvers package can be imported and exposes the idaklu module.

        The idaklu module is the core C++ extension that provides SUNDIALS IDA solver
        bindings with KLU sparse linear solver support.
        """
        import pybammsolvers

        assert hasattr(pybammsolvers, "idaklu")

        # Ensure the idaklu C++ extension module is importable
        import pybammsolvers.idaklu

        assert pybammsolvers.idaklu is not None

        # Versioning
        assert hasattr(pybammsolvers, "__version__")
        assert isinstance(pybammsolvers.__version__, str)
        assert len(pybammsolvers.__version__) > 0

    def test_idaklu_module_attributes(self, idaklu_module):
        """
        Verify idaklu module exposes the expected classes and functions.

        The module should provide Solution, VectorNdArray, VectorSolution for managing
        solver results, and IdakluJax for JAX integration.
        """

        # Core solution classes
        assert hasattr(idaklu_module, "solution")
        assert hasattr(idaklu_module, "VectorNdArray")
        assert hasattr(idaklu_module, "VectorSolution")
        assert hasattr(idaklu_module, "VectorRealtypeNdArray")

        # Solver classes
        assert hasattr(idaklu_module, "IDAKLUSolverGroup")
        assert hasattr(idaklu_module, "StandaloneNewtonSolver")

        # Core functions
        assert hasattr(idaklu_module, "create_casadi_solver_group")
        assert hasattr(idaklu_module, "generate_function")
        assert hasattr(idaklu_module, "observe")
        assert hasattr(idaklu_module, "observe_hermite_interp")

        # JAX integration
        assert hasattr(idaklu_module, "IdakluJax")
        assert hasattr(idaklu_module, "create_idaklu_jax")
        assert hasattr(idaklu_module, "registrations")

        # Verify key classes and functions are callable/instantiable
        assert callable(idaklu_module.solution)
        assert callable(idaklu_module.VectorNdArray)
        assert callable(idaklu_module.VectorSolution)
        assert callable(idaklu_module.VectorRealtypeNdArray)
        assert callable(idaklu_module.IDAKLUSolverGroup)
        assert callable(idaklu_module.StandaloneNewtonSolver)
        assert callable(idaklu_module.create_casadi_solver_group)
        assert callable(idaklu_module.generate_function)
        assert callable(idaklu_module.create_idaklu_jax)

    def test_solution_holds_no_sensitivity_derivatives(self, idaklu_module):
        """
        Verify the solution does not carry the time derivatives of the
        sensitivities, which no consumer reads and which would cost
        n_params * n_steps * n_states values per solve.
        """
        assert hasattr(idaklu_module.solution, "yS")
        assert not hasattr(idaklu_module.solution, "ypS")

    @pytest.mark.parametrize(
        ("flag", "name"),
        [
            (-9, "IDA_REP_RES_ERR"),
            (-41, "IDA_SRES_FAIL"),
            (-42, "IDA_REP_SRES_ERR"),
        ],
    )
    def test_residual_failure_messages(self, idaklu_module, flag, name):
        """
        Verify the residual failures the solver can raise have named messages,
        as non-finite residuals fail recoverably until IDAS gives up.
        """
        message = idaklu_module.sundials_error_message(flag)
        assert message.startswith(f"{name}: ")
