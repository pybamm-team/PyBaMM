import numpy as np

import pybamm


class TestBasicSPMWith3DThermal:
    def test_spm_3d_vs_lumped_box(self):
        lumped_model = pybamm.lithium_ion.SPM(options={"thermal": "lumped"})

        three_d_model = pybamm.lithium_ion.BasicSPM_with_3DThermal(
            options={"cell geometry": "box", "dimensionality": 3}
        )

        params = pybamm.ParameterValues("Chen2020")

        t_ramp = np.array([0, 1, 3600])
        I_ramp = np.array([0, 5, 5])
        current_func = pybamm.Interpolant(t_ramp, I_ramp, pybamm.t)

        params.update(
            {
                "Current function [A]": current_func,
                "Ambient temperature [K]": 298.15,
                "Initial temperature [K]": 298.15,
                "Total heat transfer coefficient [W.m-2.K-1]": 100,
                "Left face heat transfer coefficient [W.m-2.K-1]": 100,
                "Right face heat transfer coefficient [W.m-2.K-1]": 100,
                "Front face heat transfer coefficient [W.m-2.K-1]": 100,
                "Back face heat transfer coefficient [W.m-2.K-1]": 100,
                "Bottom face heat transfer coefficient [W.m-2.K-1]": 100,
                "Top face heat transfer coefficient [W.m-2.K-1]": 100,
                # Use high thermal conductivity to make models comparable
                "Negative electrode thermal conductivity [W.m-1.K-1]": 1e5,
                "Separator thermal conductivity [W.m-1.K-1]": 1e5,
                "Positive electrode thermal conductivity [W.m-1.K-1]": 1e5,
                "Negative current collector thermal conductivity [W.m-1.K-1]": 1e5,
                "Positive current collector thermal conductivity [W.m-1.K-1]": 1e5,
            },
            check_already_exists=False,
        )
        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 20,
            "r_p": 20,
            "y": None,
            "z": None,
            "x": None,
        }
        lumped_sim = pybamm.Simulation(lumped_model, parameter_values=params)
        three_d_sim = pybamm.Simulation(
            three_d_model, parameter_values=params, var_pts=var_pts
        )

        lumped_sol = lumped_sim.solve([0, 3600])
        three_d_sol = three_d_sim.solve([0, 3600])

        np.testing.assert_allclose(lumped_sol.t[-1], three_d_sol.t[-1], rtol=0.01)

        lumped_temp_final = lumped_sol["Volume-averaged cell temperature [K]"].entries[
            -1
        ]
        three_d_temp_final = three_d_sol[
            "Volume-averaged cell temperature [K]"
        ].entries[-1]
        np.testing.assert_allclose(lumped_temp_final, three_d_temp_final, rtol=0.01)

        lumped_volt_final = lumped_sol["Voltage [V]"].entries[-1]
        three_d_volt_final = three_d_sol["Voltage [V]"].entries[-1]
        np.testing.assert_allclose(lumped_volt_final, three_d_volt_final, rtol=0.01)

    def test_spm_3d_vs_lumped_cylinder(self):
        lumped_model = pybamm.lithium_ion.SPM(options={"thermal": "lumped"})

        three_d_model = pybamm.lithium_ion.BasicSPM_with_3DThermal(
            options={"cell geometry": "cylindrical", "dimensionality": 3}
        )

        params = pybamm.ParameterValues("NCA_Kim2011")

        capacity = params["Nominal cell capacity [A.h]"]
        params.update(
            {
                "Current function [A]": capacity,
                "Ambient temperature [K]": 298.15,
                "Initial temperature [K]": 298.15,
                "Total heat transfer coefficient [W.m-2.K-1]": 100,
                "Inner cell radius [m]": 0.005,
                "Outer cell radius [m]": 0.018,
                "Inner radius heat transfer coefficient [W.m-2.K-1]": 100,
                "Outer radius heat transfer coefficient [W.m-2.K-1]": 100,
                "Bottom face heat transfer coefficient [W.m-2.K-1]": 100,
                "Top face heat transfer coefficient [W.m-2.K-1]": 100,
                "Negative electrode thermal conductivity [W.m-1.K-1]": 1e5,
                "Separator thermal conductivity [W.m-1.K-1]": 1e5,
                "Positive electrode thermal conductivity [W.m-1.K-1]": 1e5,
                "Negative current collector thermal conductivity [W.m-1.K-1]": 1e5,
                "Positive current collector thermal conductivity [W.m-1.K-1]": 1e5,
            },
            check_already_exists=False,
        )
        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 20,
            "r_p": 20,
            "y": None,
            "z": None,
            "x": None,
            "r_macro": None,
        }
        lumped_sim = pybamm.Simulation(lumped_model, parameter_values=params)
        three_d_sim = pybamm.Simulation(
            three_d_model, parameter_values=params, var_pts=var_pts
        )

        lumped_sol = lumped_sim.solve([0, 3600])
        three_d_sol = three_d_sim.solve([0, 3600])

        np.testing.assert_allclose(lumped_sol.t[-1], three_d_sol.t[-1], rtol=0.01)

        lumped_temp_final = lumped_sol["Volume-averaged cell temperature [K]"].entries[
            -1
        ]
        three_d_temp_final = three_d_sol[
            "Volume-averaged cell temperature [K]"
        ].entries[-1]
        np.testing.assert_allclose(lumped_temp_final, three_d_temp_final, rtol=0.01)

        lumped_volt_final = lumped_sol["Voltage [V]"].entries[-1]
        three_d_volt_final = three_d_sol["Voltage [V]"].entries[-1]
        np.testing.assert_allclose(lumped_volt_final, three_d_volt_final, rtol=0.01)
