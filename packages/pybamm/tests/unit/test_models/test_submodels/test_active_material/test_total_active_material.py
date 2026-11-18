import numpy as np

import pybamm

PHASE_PARAMETERS = [
    "Negative electrode OCP [V]",
    "Negative electrode OCP entropic change [V.K-1]",
    "Maximum concentration in negative electrode [mol.m-3]",
    "Initial concentration in negative electrode [mol.m-3]",
    "Negative particle radius [m]",
    "Negative particle diffusivity [m2.s-1]",
    "Negative electrode exchange-current density [A.m-2]",
    "Negative electrode active material volume fraction",
]


def two_phase_parameter_values(eps_1, R_1, eps_2, R_2):
    parameter_values = pybamm.ParameterValues("Chen2020")
    for name in PHASE_PARAMETERS:
        value = parameter_values[name]
        parameter_values.update(
            {f"Primary: {name}": value, f"Secondary: {name}": value},
            check_already_exists=False,
        )
        del parameter_values[name]
    parameter_values.update(
        {
            "Primary: Negative electrode active material volume fraction": eps_1,
            "Primary: Negative particle radius [m]": R_1,
            "Secondary: Negative electrode active material volume fraction": eps_2,
            "Secondary: Negative particle radius [m]": R_2,
        }
    )
    return parameter_values


def evaluate_variable(model, parameter_values, name):
    geometry = model.default_geometry
    parameter_values.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    symbol = parameter_values.process_symbol(model.variables[name])
    return disc.process_symbol(symbol).evaluate()


class TestTotalActiveMaterial:
    def test_surface_area_to_volume_ratio_is_sum_over_phases(self):
        eps_1, R_1, eps_2, R_2 = 0.5, 5.86e-6, 0.25, 2e-6
        parameter_values = two_phase_parameter_values(eps_1, R_1, eps_2, R_2)
        model = pybamm.lithium_ion.SPM({"particle phases": ("2", "1")})

        for prefix in ["Negative", "X-averaged negative"]:
            a = evaluate_variable(
                model,
                parameter_values,
                f"{prefix} electrode surface area to volume ratio [m-1]",
            )
            np.testing.assert_allclose(a, 3 * eps_1 / R_1 + 3 * eps_2 / R_2, rtol=1e-12)

    def test_splitting_into_identical_phases_keeps_surface_area_to_volume_ratio(
        self,
    ):
        single_phase_values = pybamm.ParameterValues("Chen2020")
        eps = single_phase_values["Negative electrode active material volume fraction"]
        R = single_phase_values["Negative particle radius [m]"]
        name = "X-averaged negative electrode surface area to volume ratio [m-1]"

        a_single = evaluate_variable(
            pybamm.lithium_ion.SPM(), single_phase_values, name
        )
        a_split = evaluate_variable(
            pybamm.lithium_ion.SPM({"particle phases": ("2", "1")}),
            two_phase_parameter_values(eps / 2, R, eps / 2, R),
            name,
        )
        np.testing.assert_allclose(a_split, a_single, rtol=1e-12)
        np.testing.assert_allclose(a_split, 3 * eps / R, rtol=1e-12)
