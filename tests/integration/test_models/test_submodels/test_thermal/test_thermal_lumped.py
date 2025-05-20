import numpy as np

import pybamm


class TestLumpedThermalModel:
    def test_lumped_model_matches_standard(self):
        options_standard = {"thermal": "lumped"}
        model_standard = pybamm.lithium_ion.SPM(options=options_standard)
        param_standard = pybamm.ParameterValues("Chen2020")

        options_lumped = {"thermal": "lumped", "use lumped thermal capacity": "true"}
        model_lumped = pybamm.lithium_ion.SPM(options=options_lumped)
        param_lumped = pybamm.ParameterValues("Chen2020")

        sim_standard = pybamm.Simulation(
            model_standard, parameter_values=param_standard
        )
        solution_standard = sim_standard.solve([0, 3600])

        heat_capacity_standard = solution_standard[
            "Volume-averaged effective heat capacity [J.K-1.m-3]"
        ].data[0]

        param_lumped.update(
            {
                "Cell heat capacity [J.K-1.m-3]": heat_capacity_standard,
            },
            check_already_exists=False,
        )

        sim_lumped = pybamm.Simulation(model_lumped, parameter_values=param_lumped)
        solution_lumped = sim_lumped.solve([0, 3600])

        T_standard = solution_standard["Volume-averaged cell temperature [K]"].data
        T_lumped = solution_lumped["Volume-averaged cell temperature [K]"].data

        np.testing.assert_allclose(T_standard, T_lumped, rtol=1e-3)
