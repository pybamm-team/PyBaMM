from __future__ import annotations

import numpy as np
import pytest

import pybamm


class TestElectrolyteMigrationFlux:
    @pytest.mark.parametrize("points", [10, 20, 40])
    @pytest.mark.parametrize("graded", [False, True])
    @pytest.mark.parametrize("order", ["linear", "quadratic"])
    def test_variable_transference_conserves_lithium(self, graded, order, points):
        model = pybamm.lithium_ion.SPMe()
        parameters = pybamm.ParameterValues("ORegan2022")
        parameters.update({"Current function [A]": 2.5})
        submeshes = model.default_submesh_types
        if graded:
            for domain in model.options.whole_cell_domains:
                submeshes[domain] = pybamm.MeshGenerator(
                    pybamm.Exponential1DSubMesh,
                    submesh_params={"side": "left", "stretch": 2.3},
                )
        methods = model.default_spatial_methods
        methods["macroscale"] = pybamm.FiniteVolume(
            {"extrapolation": {"order": {"value": order, "gradient": "quadratic"}}}
        )
        simulation = pybamm.Simulation(
            model,
            parameter_values=parameters,
            var_pts={"x_n": points, "x_s": points, "x_p": points, "r_n": 30, "r_p": 30},
            submesh_types=submeshes,
            spatial_methods=methods,
            solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-11),
        )
        solution = simulation.solve([0, 600])
        assert solution.t[-1] == pytest.approx(600)
        edges = simulation.mesh[model.options.whole_cell_domains].edges
        flux = solution["Electrolyte flux [mol.m-2.s-1]"](solution.t, x=edges[[0, -1]])
        np.testing.assert_allclose(flux, 0, rtol=0, atol=1e-15)
        for variable in ["Total lithium in electrolyte [mol]", "Total lithium [mol]"]:
            inventory = solution[variable](solution.t)
            np.testing.assert_allclose(inventory, inventory[0], rtol=0, atol=1e-12)
