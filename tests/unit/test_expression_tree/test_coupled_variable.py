#
# Tests for the CoupledVariable class
#

import pytest

import numpy as np

import pybamm

def combine_models(list_of_models):
    model = pybamm.BaseModel()
    
    for submodel in list_of_models:
        model.coupled_variables.update(submodel.coupled_variables)
        model.variables.update(submodel.variables)
        model.rhs.update(submodel.rhs)
        model.algebraic.update(submodel.algebraic)
        model.initial_conditions.update(submodel.initial_conditions)
        model.boundary_conditions.update(submodel.boundary_conditions)
    
    for name, coupled_variable in model.coupled_variables.items():
        if name in model.variables:
            for sym in model.rhs.values():
                coupled_variable.set_coupled_variable(sym, model.variables[name])
            for sym in model.algebraic.values():
                coupled_variable.set_coupled_variable(sym, model.variables[name])
    return model


class TestCoupledVariable:
    def test_coupled_variable(self):
        model_1 = pybamm.BaseModel()
        model_1_var_1 = pybamm.CoupledVariable("a")
        model_1_var_2 = pybamm.Variable("b")
        model_1.rhs[model_1_var_2] = -0.2 * model_1_var_1
        model_1.variables["b"] = model_1_var_2
        model_1.coupled_variables["a"] = model_1_var_1
        model_1.initial_conditions[model_1_var_2] = 1.0

        model_2 = pybamm.BaseModel()
        model_2_var_1 = pybamm.Variable("a")
        model_2_var_2 = pybamm.CoupledVariable("b")
        model_2.rhs[model_2_var_1] = - 0.2 * model_2_var_2
        model_2.variables["a"] = model_2_var_1
        model_2.coupled_variables["b"] = model_2_var_2
        model_2.initial_conditions[model_2_var_1] = 1.0

        model = combine_models([model_1, model_2])

        params = pybamm.ParameterValues({})
        geometry = {}

        # Process parameters
        params.process_model(model)
        params.process_geometry(geometry)

        # mesh and discretise
        submesh_types = {}
        var_pts = {}
        mesh = pybamm.Mesh(geometry, submesh_types, var_pts)


        spatial_methods = {}
        disc = pybamm.Discretisation(mesh, spatial_methods)
        disc.process_model(model)

        # solve
        solver = pybamm.CasadiSolver()
        t = np.linspace(0, 10, 1000)
        solution = solver.solve(model, t)

        np.testing.assert_almost_equal(solution["a"].entries, solution["b"].entries, decimal=10)



