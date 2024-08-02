#
# Tests for the thermal lithium-ion models produce consistent
# thermal response
#
import pybamm
import numpy as np
import unittest


class TestThermal(unittest.TestCase):
    def test_consistent_cooling(self):
        "Test the cooling is consistent between the 1D, 1+1D and 2+1D SPMe models"

        # Load models
        options = {"thermal": "lumped"}
        spme_1D = pybamm.lithium_ion.SPMe(options=options)
        options = {
            "thermal": "x-lumped",
            "current collector": "potential pair",
            "dimensionality": 1,
        }
        spme_1p1D = pybamm.lithium_ion.SPMe(options=options)
        options = {
            "thermal": "x-lumped",
            "current collector": "potential pair",
            "dimensionality": 2,
        }
        spme_2p1D = pybamm.lithium_ion.SPMe(options=options)
        models = {"SPMe 1D": spme_1D, "SPMe 1+1D": spme_1p1D, "SPMe 2+1D": spme_2p1D}

        # Set up parameter values
        parameter_values = pybamm.ParameterValues("NCA_Kim2011")
        C_rate = 5
        h_cn = 10
        h_cp = 5
        h_tab_n = 250
        h_tab_p = 250
        h_edge = 100
        # for the lumped model, the total heat transfer coefficient is the area-weighted
        # average of the heat transfer coefficients
        param = spme_1D.param
        L = param.L
        L_y = param.L_y
        L_z = param.L_z
        L_tab_n = param.n.L_tab
        L_tab_p = param.p.L_tab
        L_cn = param.n.L_cc
        L_cp = param.p.L_cc

        h_total = (
            h_cn * L_y * L_z
            + h_cp * L_y * L_z
            + h_tab_n * L_tab_n * L_cn
            + h_tab_p * L_tab_p * L_cp
            + h_edge * (2 * L_y * L + 2 * L_z * L - L_tab_n * L_cn - L_tab_p * L_cp)
        ) / (2 * L_y * L_z + 2 * L_y * L + 2 * L_z * L)

        parameter_values.update(
            {
                "Negative current collector"
                + " surface heat transfer coefficient [W.m-2.K-1]": h_cn,
                "Positive current collector"
                + " surface heat transfer coefficient [W.m-2.K-1]": h_cp,
                "Negative tab heat transfer coefficient [W.m-2.K-1]": h_tab_n,
                "Positive tab heat transfer coefficient [W.m-2.K-1]": h_tab_p,
                "Edge heat transfer coefficient [W.m-2.K-1]": h_edge,
                "Total heat transfer coefficient [W.m-2.K-1]": h_total,
                # Set high thermal and electrical conductivity in current collectors
                "Negative current collector"
                + " thermal conductivity [W.m-1.K-1]": 267.467 * 100000,
                "Positive current collector"
                + " thermal conductivity [W.m-1.K-1]": 158.079 * 100000,
                "Negative current collector conductivity [S.m-1]": 1e10,
                "Positive current collector conductivity [S.m-1]": 1e10,
            }
        )

        # Solve models
        solutions = {}
        var_pts = {"x_n": 3, "x_s": 3, "x_p": 3, "r_n": 3, "r_p": 3, "y": 5, "z": 5}
        for model_name, model in models.items():
            solver = pybamm.CasadiSolver(mode="fast")
            sim = pybamm.Simulation(
                model,
                var_pts=var_pts,
                solver=solver,
                parameter_values=parameter_values,
                C_rate=C_rate,
            )
            t_eval = np.linspace(0, 3500 / 6, 100)
            sim.solve(t_eval=t_eval)

            solutions[model_name] = sim.solution[
                "Volume-averaged cell temperature [K]"
            ].entries

        # check volume-averaged cell temperature is within
        # 1e-5 relative error

        def err(a, b):
            return np.max(np.abs(a - b)) / np.max(np.abs(a))

        self.assertGreater(1e-5, err(solutions["SPMe 1+1D"], solutions["SPMe 2+1D"]))

    def test_lumped_contact_resistance(self):
        # Test that the heating with contact resistance is greater than without

        # load models
        model_no_contact_resistance = pybamm.lithium_ion.SPMe(
            {
                "cell geometry": "arbitrary",
                "thermal": "lumped",
                "contact resistance": "false",
            }
        )
        model_contact_resistance = pybamm.lithium_ion.SPMe(
            {
                "cell geometry": "arbitrary",
                "thermal": "lumped",
                "contact resistance": "true",
            }
        )
        models = [model_no_contact_resistance, model_contact_resistance]

        # parameters
        parameter_values = pybamm.ParameterValues("Marquis2019")
        lumped_params = parameter_values.copy()
        lumped_params_contact_resistance = parameter_values.copy()

        lumped_params_contact_resistance.update(
            {
                "Contact resistance [Ohm]": 0.05,
            }
        )

        # solve the models
        params = [lumped_params, lumped_params_contact_resistance]
        sols = []
        for model, param in zip(models, params):
            sim = pybamm.Simulation(model, parameter_values=param)
            sim.solve([0, 3600])
            sols.append(sim.solution)

        # get the average temperature from each model
        avg_cell_temp = sols[0]["X-averaged cell temperature [K]"].entries
        avg_cell_temp_cr = sols[1]["X-averaged cell temperature [K]"].entries

        # check that the cell temperature of the lumped thermal model
        # with contact resistance is higher than without contact resistance
        # skip the first entry because they are the same due to initial conditions
        np.testing.assert_array_less(avg_cell_temp[1:], avg_cell_temp_cr[1:])


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
