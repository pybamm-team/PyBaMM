from __future__ import annotations

import numpy as np
import pytest

import pybamm


class TestSEIGrowth:
    @pytest.mark.parametrize(
        "model_class",
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ids=["spm", "dfn"],
    )
    @pytest.mark.parametrize("temperature", [298.15, 318.15])
    def test_solvent_diffusion_growth_across_restart(self, model_class, temperature):
        parameter_values = pybamm.ParameterValues("OKane2022")
        parameter_values.update(
            {
                "Ambient temperature [K]": temperature,
                "Initial temperature [K]": temperature,
                # A non-unit stoichiometry exposes missing conversion factors.
                "Ratio of lithium moles to SEI moles": 2.0,
            }
        )
        solution = None
        for steps in [
            ["Rest for 12 hours"],
            [("Discharge at C/10 for 10 minutes", "Rest for 12 hours")],
        ]:
            simulation = pybamm.Simulation(
                model_class(
                    {"SEI": "solvent-diffusion limited", "thermal": "isothermal"}
                ),
                parameter_values=parameter_values,
                experiment=pybamm.Experiment(steps),
                # Resolve small SEI increments relative to the initial film thickness.
                solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-8),
            )
            solution = simulation.solve(starting_solution=solution, calc_esoh=False)

        times = np.array([0, 21600, 43200, 43500, 43800, 65400, 87000])
        assert solution.t[-1] == pytest.approx(times[-1])
        initial_thickness = parameter_values["Initial SEI thickness [m]"]
        molar_volume = parameter_values["SEI partial molar volume [m3.mol-1]"]
        stoichiometry = parameter_values["Ratio of lithium moles to SEI moles"]
        arrhenius = np.exp(
            parameter_values["SEI growth activation energy [J.mol-1]"]
            / pybamm.constants.R.value
            * (1 / parameter_values["Reference temperature [K]"] - 1 / temperature)
        )
        # At fixed temperature and reaction area, d(L**2)/dt is constant.
        growth_rate = (
            2
            * molar_volume
            * parameter_values["SEI solvent diffusivity [m2.s-1]"]
            * parameter_values["Bulk solvent concentration [mol.m-3]"]
            * arrhenius
            / stoichiometry
        )
        expected_thickness = np.sqrt(initial_thickness**2 + growth_rate * times)
        np.testing.assert_allclose(
            solution["X-averaged negative SEI thickness [m]"](times),
            expected_thickness,
            rtol=1e-5,
            atol=1e-14,
        )

        surface_area = (
            parameter_values["Electrode width [m]"]
            * parameter_values["Electrode height [m]"]
            * parameter_values["Negative electrode thickness [m]"]
            * 3
            * parameter_values["Negative electrode active material volume fraction"]
            / parameter_values["Negative particle radius [m]"]
        )
        expected_lithium_loss = (
            stoichiometry
            * surface_area
            * (expected_thickness - initial_thickness)
            / molar_volume
        )
        np.testing.assert_allclose(
            solution["Loss of lithium to negative SEI [mol]"](times),
            expected_lithium_loss,
            rtol=1e-4,
            atol=1e-10,
        )
