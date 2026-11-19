import pytest

import pybamm


class TestHalfCellSurfaceForm:
    def test_phi_e_boundary_conditions(self):
        model = pybamm.lithium_ion.DFN(
            {"working electrode": "positive", "surface form": "algebraic"}
        )
        phi_e = model.variables["Electrolyte potential [V]"]
        assert phi_e in model.boundary_conditions

    def test_phi_s_boundary_conditions(self):
        model = pybamm.lithium_ion.DFN(
            {"working electrode": "positive", "surface form": "algebraic"}
        )
        phi_s_p = model.variables["Positive electrode potential [V]"]
        assert phi_s_p in model.boundary_conditions

    @pytest.mark.parametrize("surface_form", ["algebraic", "differential"])
    def test_lumped_thermal_builds(self, surface_form):
        # Xu2019 has no current collector or thermal parameters, so take them from
        # Chen2020
        parameter_values = pybamm.ParameterValues("Xu2019")
        chen2020 = pybamm.ParameterValues("Chen2020")
        parameter_values.update(
            {k: chen2020[k] for k in chen2020 if k not in parameter_values},
            check_already_exists=False,
        )
        model = pybamm.lithium_ion.DFN(
            {
                "working electrode": "positive",
                "surface form": surface_form,
                "thermal": "lumped",
            }
        )
        pybamm.Simulation(model, parameter_values=parameter_values).build()
