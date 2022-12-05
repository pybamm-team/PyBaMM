#
# Tests for external submodels
#
import pybamm
import unittest
import numpy as np


class TestExternalThermalModels(unittest.TestCase):
    def test_input_lumped_temperature(self):
        model = pybamm.lithium_ion.SPMe()
        parameter_values = model.default_parameter_values
        # in the default isothermal model, the temperature is everywhere equal
        # to the ambient temperature
        parameter_values["Ambient temperature [K]"] = pybamm.InputParameter(
            "Volume-averaged cell temperature [K]"
        )
        sim = pybamm.Simulation(model)

        t_eval = np.linspace(0, 100, 3)

        T_av = 298

        for i in np.arange(1, len(t_eval) - 1):
            dt = t_eval[i + 1] - t_eval[i]
            inputs = {"Volume-averaged cell temperature [K]": T_av}
            T_av += 1
            sim.step(dt, inputs=inputs)  # works

    # def test_external_temperature(self):

    #     model = pybamm.lithium_ion.SPM()

    #     neg_pts = 5
    #     sep_pts = 3
    #     pos_pts = 5
    #     tot_pts = neg_pts + sep_pts + pos_pts

    #     var_pts = {
    #         pybamm.standard_spatial_vars.x_n: neg_pts,
    #         pybamm.standard_spatial_vars.x_s: sep_pts,
    #         pybamm.standard_spatial_vars.x_p: pos_pts,
    #         pybamm.standard_spatial_vars.r_n: 5,
    #         pybamm.standard_spatial_vars.r_p: 5,
    #     }

    #     parameter_values = model.default_parameter_values
    #     # in the default isothermal model, the temperature is everywhere equal
    #     # to the ambient temperature
    #     parameter_values["Ambient temperature [K]"] = pybamm.InputParameter(
    #         "Cell temperature [K]",
    #         domain=["negative electrode", "separator", "positive electrode"],
    #     )
    #     sim = pybamm.Simulation(
    #         model, var_pts=var_pts, parameter_values=parameter_values
    #     )

    #     t_eval = np.linspace(0, 100, 3)
    #     x = np.linspace(0, 1, tot_pts)

    #     for i in np.arange(1, len(t_eval) - 1):
    #         dt = t_eval[i + 1] - t_eval[i]
    #         T = (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * 100 * t_eval[i]))[
    #             :, np.newaxis
    #         ]
    #         inputs = {"Cell temperature [K]": T}
    #         sim.step(dt, inputs=inputs)  # fails because of broadcasting error

    # def test_dae_external_temperature(self):

    #     model_options = {"thermal": "x-full", "external submodels": ["thermal"]}

    #     model = pybamm.lithium_ion.DFN(model_options)

    #     neg_pts = 5
    #     sep_pts = 3
    #     pos_pts = 5
    #     tot_pts = neg_pts + sep_pts + pos_pts

    #     var_pts = {
    #         pybamm.standard_spatial_vars.x_n: neg_pts,
    #         pybamm.standard_spatial_vars.x_s: sep_pts,
    #         pybamm.standard_spatial_vars.x_p: pos_pts,
    #         pybamm.standard_spatial_vars.r_n: 5,
    #         pybamm.standard_spatial_vars.r_p: 5,
    #     }

    #     solver = pybamm.CasadiSolver()
    #     sim = pybamm.Simulation(model, var_pts=var_pts, solver=solver)
    #     sim.build()

    #     t_eval = np.linspace(0, 100, 3)
    #     x = np.linspace(0, 1, tot_pts)

    #     for i in np.arange(1, len(t_eval) - 1):
    #         dt = t_eval[i + 1] - t_eval[i]
    #         T = (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * 100 * t_eval[i]))[
    #             :, np.newaxis
    #         ]
    #         external_variables = {"Cell temperature": T}
    #         sim.step(dt, external_variables=external_variables)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
