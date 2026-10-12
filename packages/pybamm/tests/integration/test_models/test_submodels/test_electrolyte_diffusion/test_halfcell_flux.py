from __future__ import annotations

import numpy as np
import pytest

import pybamm


class GradedSpectralVolumeMesh(pybamm.SpectralVolume1DSubMesh):
    def __init__(self, lims, npts, order=2):
        spatial_var, bounds, _ = self.read_lims(lims)
        unit_edges = np.linspace(0, 1, npts[spatial_var.name] + 1)
        unit_edges = np.expm1(2.3 * unit_edges) / np.expm1(2.3)
        edges = bounds["min"] + (bounds["max"] - bounds["min"]) * unit_edges
        edges[0], edges[-1] = bounds["min"], bounds["max"]
        super().__init__(lims, npts, edges=edges, order=order)


def make_simulation(method, points, graded, value_order, parameters):
    model = pybamm.lithium_ion.SPMe({"working electrode": "positive"})
    methods = model.default_spatial_methods
    submeshes = model.default_submesh_types
    options = {
        "extrapolation": {"order": {"value": value_order, "gradient": "quadratic"}}
    }
    if method == "FV":
        methods["macroscale"] = pybamm.FiniteVolume(options)
        if graded:
            for domain in model.options.whole_cell_domains:
                submeshes[domain] = pybamm.MeshGenerator(
                    pybamm.Exponential1DSubMesh,
                    {"side": "left", "stretch": 2.3},
                )
    else:
        degree = int(method[-1])
        methods["macroscale"] = pybamm.SpectralVolume(options, order=degree)
        for domain in model.options.whole_cell_domains:
            submeshes[domain] = pybamm.MeshGenerator(
                GradedSpectralVolumeMesh if graded else pybamm.SpectralVolume1DSubMesh,
                {"order": degree},
            )
    return pybamm.Simulation(
        model,
        parameter_values=parameters,
        spatial_methods=methods,
        submesh_types=submeshes,
        var_pts={"x_n": points, "x_s": points, "x_p": points, "r_n": 30, "r_p": 30},
        solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-11),
    )


class TestHalfCellElectrolyteFlux:
    @pytest.mark.parametrize("method", ["FV", "SV2", "SV3"])
    @pytest.mark.parametrize("points", [10, 20])
    @pytest.mark.parametrize("graded", [False, True])
    @pytest.mark.parametrize("value_order", ["linear", "quadratic"])
    def test_concentration_dependent_diffusion_conserves_lithium(
        self, method, points, graded, value_order
    ):
        parameters = pybamm.ParameterValues("OKane2022_graphite_SiOx_halfcell")
        parameters.update({"Current function [A]": -2.5})
        sim = make_simulation(method, points, graded, value_order, parameters)
        sol = sim.solve([0, 600])
        assert sol.t[-1] == pytest.approx(600)
        inventory = sol["Total lithium in electrolyte [mol]"](sol.t)
        np.testing.assert_allclose(inventory, inventory[0], rtol=0, atol=1e-11)
        closed_inventory = (
            sol["Total lithium [mol]"](sol.t)
            - sol["Discharge capacity [A.h]"](sol.t) * 3600 / pybamm.constants.F.value
        )
        np.testing.assert_allclose(
            closed_inventory, closed_inventory[0], rtol=0, atol=1e-11
        )

    @pytest.mark.parametrize("method", ["FV", "SV2", "SV3"])
    @pytest.mark.parametrize("value_order", ["linear", "quadratic"])
    def test_concentration_dependent_transference_boundary_flux(
        self, method, value_order
    ):
        parameters = pybamm.ParameterValues("OKane2022_graphite_SiOx_halfcell")
        parameters.update(
            {
                "Current function [A]": -2.5,
                "Cation transference number": pybamm.ParameterValues("ORegan2022")[
                    "Cation transference number"
                ],
            }
        )
        sim = make_simulation(method, 10, True, value_order, parameters)
        sol = sim.solve([0, 600])
        left = sim.mesh["separator"].edges[0]
        flux = sol["Electrolyte flux [mol.m-2.s-1]"](sol.t, x=left)
        area = parameters["Electrode height [m]"] * parameters["Electrode width [m]"]
        expected = -2.5 / (area * pybamm.constants.F.value)
        np.testing.assert_allclose(flux, expected, rtol=1e-10, atol=1e-15)

    @pytest.mark.parametrize("method", ["FV", "SV2", "SV3"])
    def test_constant_diffusivity_control(self, method):
        parameters = pybamm.ParameterValues("OKane2022_graphite_SiOx_halfcell")
        parameters.update(
            {"Current function [A]": -2.5, "Electrolyte diffusivity [m2.s-1]": 2e-10}
        )
        sim = make_simulation(method, 10, True, "linear", parameters)
        sol = sim.solve([0, 600])
        inventory = sol["Total lithium in electrolyte [mol]"](sol.t)
        np.testing.assert_allclose(inventory, inventory[0], rtol=0, atol=1e-11)

    @pytest.mark.parametrize("graded", [False, True])
    def test_dfn_concentration_dependent_diffusion(self, graded):
        model = pybamm.lithium_ion.DFN({"working electrode": "positive"})
        parameters = pybamm.ParameterValues("OKane2022_graphite_SiOx_halfcell")
        parameters.update({"Current function [A]": -2.5})
        submeshes = model.default_submesh_types
        if graded:
            for domain in model.options.whole_cell_domains:
                submeshes[domain] = pybamm.MeshGenerator(
                    pybamm.Exponential1DSubMesh,
                    {"side": "left", "stretch": 2.3},
                )
        sim = pybamm.Simulation(
            model,
            parameter_values=parameters,
            var_pts={"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 30, "r_p": 30},
            submesh_types=submeshes,
            solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-11),
        )
        sol = sim.solve([0, 600])
        assert sol.t[-1] == pytest.approx(600)
        closed_inventory = (
            sol["Total lithium [mol]"](sol.t)
            - sol["Discharge capacity [A.h]"](sol.t) * 3600 / pybamm.constants.F.value
        )
        np.testing.assert_allclose(
            closed_inventory, closed_inventory[0], rtol=0, atol=1e-11
        )
