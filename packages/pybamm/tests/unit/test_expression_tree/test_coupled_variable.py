#
# Tests for the CoupledVariable class
#


import numpy as np
import pytest

import pybamm


def combine_models(list_of_models):
    """
    Combine multiple models into one.

    With lazy resolution of CoupledVariables, we no longer need to manually
    call set_coupled_variable. The discretisation will automatically resolve
    CoupledVariables by looking them up in model.variables.
    """
    model = pybamm.BaseModel()

    for submodel in list_of_models:
        model.coupled_variables.update(submodel.coupled_variables)
        model.variables.update(submodel.variables)
        model.rhs.update(submodel.rhs)
        model.algebraic.update(submodel.algebraic)
        model.initial_conditions.update(submodel.initial_conditions)
        model.boundary_conditions.update(submodel.boundary_conditions)

    # No manual linking needed - CoupledVariables are resolved lazily
    # during discretisation by looking up names in model.variables
    return model


class TestCoupledVariable:
    def test_coupled_variable(self):
        """Test combining models with CoupledVariables in rhs."""
        model_1 = pybamm.BaseModel()
        model_1_var_1 = pybamm.CoupledVariable("a")
        model_1_var_2 = pybamm.Variable("b")
        model_1.rhs[model_1_var_2] = -0.2 * model_1_var_1
        model_1.variables["b"] = model_1_var_2
        model_1.coupled_variables["a"] = model_1_var_1
        model_1.initial_conditions[model_1_var_2] = pybamm.Scalar(1)

        model_2 = pybamm.BaseModel()
        model_2_var_1 = pybamm.Variable("a")
        model_2_var_2 = pybamm.CoupledVariable("b")
        model_2.rhs[model_2_var_1] = -0.2 * model_2_var_2
        model_2.variables["a"] = model_2_var_1
        model_2.coupled_variables["b"] = model_2_var_2
        model_2.initial_conditions[model_2_var_1] = pybamm.Scalar(1)

        model = combine_models([model_1, model_2])

        disc = pybamm.Discretisation()
        disc.process_model(model)

        solver = pybamm.IDAKLUSolver()
        solution = solver.solve(model, [0, 10])

        np.testing.assert_almost_equal(
            solution["a"].entries, solution["b"].entries, decimal=10
        )

        assert set(model.list_coupled_variables()) == {"a", "b"}

    def test_create_copy(self):
        a = pybamm.CoupledVariable("a")
        b = a.create_copy()
        assert a == b
        # Takes the base class's arguments, as tree rebuilds pass them
        assert a.create_copy(new_children=None, perform_simplifications=False) == a

    def test_setter(self):
        model = pybamm.BaseModel()
        a = pybamm.CoupledVariable("a")
        coupled_variables = {"a": a}
        model.coupled_variables = coupled_variables
        assert model.coupled_variables == coupled_variables

        with pytest.raises(ValueError, match=r"Coupled variable with name"):
            coupled_variables = {"b": a}
            model.coupled_variables = coupled_variables

    def test_unresolved_coupled_variable_raises_error(self):
        """Test that an unresolved CoupledVariable raises a clear error."""
        model = pybamm.BaseModel()
        a = pybamm.Variable("a")
        unresolved = pybamm.CoupledVariable("missing_variable")

        model.rhs = {a: pybamm.Scalar(1)}
        model.initial_conditions = {a: pybamm.Scalar(1)}
        model.variables = {
            "a": a,
            "uses_missing": unresolved,
        }  # Note: "missing_variable" is NOT in variables

        sim = pybamm.Simulation(model)
        sim.solve([0, 1])

        # Error is raised when accessing the variable (during lazy processing)
        with pytest.raises(
            ValueError,
            match="CoupledVariable 'missing_variable' not found",
        ):
            _ = sim.solution["uses_missing"]

    def test_unresolved_coupled_variable_in_rhs_raises_error(self):
        """Test that a CoupledVariable in rhs raises error during discretisation."""

        def make_model():
            model = pybamm.BaseModel()
            a = pybamm.Variable("a")
            model.rhs = {a: pybamm.CoupledVariable("missing_variable")}
            model.initial_conditions = {a: pybamm.Scalar(1)}
            model.variables = {"a": a}  # "missing_variable" NOT in variables
            return model

        with pytest.raises(
            ValueError, match="CoupledVariable 'missing_variable' not found"
        ):
            pybamm.Simulation(make_model()).solve([0, 1])

        # Without resolving, the CoupledVariable hits the discretisation error
        sim = pybamm.Simulation(
            make_model(), discretisation_kwargs={"resolve_coupled_variables": False}
        )
        with pytest.raises(
            pybamm.DiscretisationError,
            match="CoupledVariable 'missing_variable' was not resolved",
        ):
            sim.solve([0, 1])

    def test_coupled_variable_with_battery_model(self):
        """Test CoupledVariable with a real battery model (SPM)."""
        model = pybamm.lithium_ion.SPM()

        # Add a variable that references an existing variable via CoupledVariable
        model.variables["Double voltage"] = 2 * pybamm.CoupledVariable("Voltage [V]")

        sim = pybamm.Simulation(model)
        sim.solve([0, 3600])

        # Verify the values are correct
        double_voltage = sim.solution["Double voltage"].entries
        voltage = sim.solution["Voltage [V]"].entries

        # Double voltage should be exactly 2x the voltage
        np.testing.assert_allclose(double_voltage, 2 * voltage, rtol=1e-10)

    def test_coupled_variable_in_boundary_condition(self):
        mesh = pybamm.Mesh(
            pybamm.Geometry(
                {
                    "negative electrode": {
                        "x_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(1)}
                    }
                }
            ),
            {"negative electrode": pybamm.Uniform1DSubMesh},
            {"x_n": 10},
        )
        spatial_methods = {"negative electrode": pybamm.FiniteVolume()}

        def make_model(flux):
            model = pybamm.BaseModel()
            c = pybamm.Variable("c", domain="negative electrode")
            model.rhs = {c: pybamm.div(pybamm.grad(c))}
            model.boundary_conditions = {
                c: {"left": (pybamm.Scalar(0), "Neumann"), "right": (flux, "Neumann")}
            }
            model.initial_conditions = {c: pybamm.Scalar(1)}
            model.variables = {"Flux": pybamm.Scalar(2), "c": c}
            return model

        solutions = []
        for flux in [pybamm.Scalar(2), pybamm.CoupledVariable("Flux")]:
            disc = pybamm.Discretisation(mesh, spatial_methods)
            model = disc.process_model(make_model(flux), inplace=False)
            solutions.append(pybamm.IDAKLUSolver().solve(model, [0, 1]))

        np.testing.assert_allclose(
            solutions[1]["c"](1.0), solutions[0]["c"](1.0), rtol=1e-10
        )

    def test_resolve_does_not_modify_model_when_not_inplace(self):
        model = pybamm.BaseModel()
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        rhs_a = -pybamm.CoupledVariable("b")
        model.rhs = {a: rhs_a, b: pybamm.Scalar(1)}
        model.initial_conditions = {a: pybamm.Scalar(1), b: pybamm.Scalar(0)}
        model.variables = {"a": a, "b": b}

        disc = pybamm.Discretisation()
        disc.process_model(model, inplace=False)

        assert model.rhs[a] is rhs_a
        assert not model.is_discretised

    def test_resolve_shares_cache_across_equations(self):
        model = pybamm.BaseModel()
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        c = pybamm.Variable("c")
        model.rhs = {
            a: pybamm.CoupledVariable("Source"),
            b: pybamm.CoupledVariable("Source"),
            c: pybamm.Scalar(1),
        }
        model.initial_conditions = {
            a: pybamm.Scalar(0),
            b: pybamm.Scalar(0),
            c: pybamm.Scalar(0),
        }
        model.variables = {
            "Source": pybamm.CoupledVariable("c") + 1,
            "a": a,
            "b": b,
            "c": c,
        }

        disc = pybamm.Discretisation()
        resolved = disc._resolve_coupled_variables_in_model(model, inplace=True)

        assert resolved is model
        assert model.rhs[a] == c + 1
        # the "Source" target is resolved once and shared between equations
        assert model.rhs[a] is model.rhs[b]

    def test_resolve_without_coupled_variables_keeps_model(self):
        model = pybamm.BaseModel()
        a = pybamm.Variable("a")
        rhs_a = -a
        model.rhs = {a: rhs_a}
        model.initial_conditions = {a: pybamm.Scalar(1)}
        model.variables = {"a": a}

        disc = pybamm.Discretisation()
        resolved = disc._resolve_coupled_variables_in_model(model, inplace=False)

        assert resolved is model
        assert model.rhs[a] is rhs_a
