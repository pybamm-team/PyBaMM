#
# Tests for the CompositeSolver class
#

import numpy as np
import pytest

import pybamm


class TestCompositeSolver:
    def test_init(self):
        solver1 = pybamm.AlgebraicSolver()
        solver2 = pybamm.CasadiAlgebraicSolver()
        composite = pybamm.CompositeSolver([solver1, solver2])
        assert composite.sub_solvers == [solver1, solver2]

    def test_init_no_solvers(self):
        with pytest.raises(ValueError, match="No sub_solvers provided"):
            pybamm.CompositeSolver([])

        with pytest.raises(ValueError, match="No sub_solvers provided"):
            pybamm.CompositeSolver(None)

    def test_first_solver_succeeds(self):
        # Create a simple algebraic model
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.algebraic = {var: var - 3}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        model.variables = {"var": var}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Both solvers can solve this, first should be used
        solver1 = pybamm.AlgebraicSolver()
        solver2 = pybamm.CasadiAlgebraicSolver()
        composite = pybamm.CompositeSolver([solver1, solver2])

        solution = composite.solve(model)
        np.testing.assert_allclose(solution.y, 3, rtol=1e-6)

    def test_fallback_to_second_solver(self):
        # Create a model that the first solver will fail on
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.algebraic = {var: var - 3}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        model.variables = {"var": var}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Create a mock solver that always fails
        class FailingSolver(pybamm.BaseSolver):
            def solve(self, *args, **kwargs):
                raise pybamm.SolverError("Intentional failure")

        failing_solver = FailingSolver()
        working_solver = pybamm.CasadiAlgebraicSolver()
        composite = pybamm.CompositeSolver([failing_solver, working_solver])

        solution = composite.solve(model)
        np.testing.assert_allclose(solution.y, 3, rtol=1e-6)

    def test_all_solvers_fail(self):
        model = pybamm.BaseModel()
        var = pybamm.Variable("var")
        model.algebraic = {var: var - 3}
        model.initial_conditions = {var: pybamm.Scalar(1)}
        model.variables = {"var": var}

        disc = pybamm.Discretisation()
        disc.process_model(model)

        # Create mock solvers that always fail
        class FailingSolver(pybamm.BaseSolver):
            def __init__(self, name):
                self.name = name

            def solve(self, *args, **kwargs):
                raise pybamm.SolverError(f"{self.name} failed")

        solver1 = FailingSolver("Solver1")
        solver2 = FailingSolver("Solver2")
        composite = pybamm.CompositeSolver([solver1, solver2])

        with pytest.raises(pybamm.SolverError, match="All sub_solvers failed"):
            composite.solve(model)
