import numpy as np

import pybamm


class TestBasicSPMWith3DThermal:
    def test_spm_3d_vs_lumped_pouch(self):
        models = {
            "Lumped": pybamm.lithium_ion.SPM(options={"thermal": "lumped"}),
            "3D": pybamm.lithium_ion.Basic3DThermalSPM(
                options={"cell geometry": "pouch", "dimensionality": 3}
            ),
        }

        parameter_values = pybamm.ParameterValues("Marquis2019")
        h_values = [0.1, 1, 10, 100]

        experiment = pybamm.Experiment(
            [
                ("Discharge at 3C until 2.8V", "Rest for 10 minutes"),
            ]
        )

        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "x": None,
            "y": None,
            "z": None,
        }

        all_solutions = {}

        for h in h_values:
            h_params = parameter_values.copy()
            h_params.update(
                {
                    "Total heat transfer coefficient [W.m-2.K-1]": h,
                    "Left face heat transfer coefficient [W.m-2.K-1]": h,
                    "Right face heat transfer coefficient [W.m-2.K-1]": h,
                    "Front face heat transfer coefficient [W.m-2.K-1]": h,
                    "Back face heat transfer coefficient [W.m-2.K-1]": h,
                    "Bottom face heat transfer coefficient [W.m-2.K-1]": h,
                    "Top face heat transfer coefficient [W.m-2.K-1]": h,
                },
                check_already_exists=False,
            )

            solutions = {}
            for model_name, model in models.items():
                sim = pybamm.Simulation(
                    model,
                    parameter_values=h_params,
                    var_pts=var_pts,
                    experiment=experiment,
                )
                solutions[model_name] = sim.solve()

            all_solutions[h] = solutions

        for _h, solutions in all_solutions.items():
            lumped_sol = solutions["Lumped"]
            three_d_sol = solutions["3D"]

            np.testing.assert_allclose(lumped_sol.t[-1], three_d_sol.t[-1], rtol=0.01)

            lumped_temp_final = lumped_sol[
                "Volume-averaged cell temperature [K]"
            ].entries[-1]
            three_d_temp_final = three_d_sol[
                "Volume-averaged cell temperature [K]"
            ].entries[-1]
            np.testing.assert_allclose(lumped_temp_final, three_d_temp_final, rtol=0.02)

            lumped_final_voltage = lumped_sol["Voltage [V]"].entries[-1]
            three_d_final_voltage = three_d_sol["Voltage [V]"].entries[-1]
            np.testing.assert_allclose(
                lumped_final_voltage, three_d_final_voltage, rtol=0.01
            )

    def test_spm_3d_vs_lumped_cylinder(self):
        models = {
            "Lumped": pybamm.lithium_ion.SPM(options={"thermal": "lumped"}),
            "3D": pybamm.lithium_ion.Basic3DThermalSPM(
                options={"cell geometry": "cylindrical", "dimensionality": 3}
            ),
        }

        parameter_values = pybamm.ParameterValues("NCA_Kim2011")
        h_values = [0.1, 1, 10, 100]

        experiment = pybamm.Experiment(
            [
                ("Discharge at 3C until 2.8V", "Rest for 10 minutes"),
            ]
        )

        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "r_macro": None,
            "y": None,
            "z": None,
        }

        all_solutions = {}

        for h in h_values:
            h_params = parameter_values.copy()
            h_params.update(
                {
                    "Inner cell radius [m]": 0.005,
                    "Outer cell radius [m]": 0.018,
                    "Total heat transfer coefficient [W.m-2.K-1]": h,
                    "Outer radius heat transfer coefficient [W.m-2.K-1]": h,
                    "Inner radius heat transfer coefficient [W.m-2.K-1]": h,
                    "Bottom face heat transfer coefficient [W.m-2.K-1]": h,
                    "Top face heat transfer coefficient [W.m-2.K-1]": h,
                },
                check_already_exists=False,
            )

            solutions = {}
            for model_name, model in models.items():
                sim = pybamm.Simulation(
                    model,
                    parameter_values=h_params,
                    var_pts=var_pts,
                    experiment=experiment,
                )
                solutions[model_name] = sim.solve()

            all_solutions[h] = solutions

        for _h, solutions in all_solutions.items():
            lumped_sol = solutions["Lumped"]
            three_d_sol = solutions["3D"]

            np.testing.assert_allclose(lumped_sol.t[-1], three_d_sol.t[-1], rtol=0.01)
            lumped_temp_final = lumped_sol[
                "Volume-averaged cell temperature [K]"
            ].entries[-1]
            three_d_temp_final = three_d_sol[
                "Volume-averaged cell temperature [K]"
            ].entries[-1]
            np.testing.assert_allclose(lumped_temp_final, three_d_temp_final, rtol=0.02)

            lumped_final_voltage = lumped_sol["Voltage [V]"].entries[-1]
            three_d_final_voltage = three_d_sol["Voltage [V]"].entries[-1]
            np.testing.assert_allclose(
                lumped_final_voltage, three_d_final_voltage, rtol=0.01
            )
