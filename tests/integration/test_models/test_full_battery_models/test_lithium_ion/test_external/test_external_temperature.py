#
# Tests for external submodels
#
import pybamm
import unittest
import numpy as np


class TestExternalThermalModels(unittest.TestCase):
    def test_external_lumped_temperature(self):
        model_options = {"thermal": "lumped", "external submodels": ["thermal"]}
        model = pybamm.lithium_ion.SPMe(model_options)
        sim = pybamm.Simulation(model)

        t_eval = np.linspace(0, 100, 3)

        T_av = 0

        for i in np.arange(1, len(t_eval) - 1):
            dt = t_eval[i + 1] - t_eval[i]
            external_variables = {"Volume-averaged cell temperature": T_av}
            T_av += 1
            sim.step(dt, external_variables=external_variables)

    def test_external_temperature(self):

        model_options = {"thermal": "x-full", "external submodels": ["thermal"]}

        model = pybamm.lithium_ion.SPM(model_options)

        neg_pts = 5
        sep_pts = 3
        pos_pts = 5
        tot_pts = neg_pts + sep_pts + pos_pts

        var_pts = {
            pybamm.standard_spatial_vars.x_n: neg_pts,
            pybamm.standard_spatial_vars.x_s: sep_pts,
            pybamm.standard_spatial_vars.x_p: pos_pts,
            pybamm.standard_spatial_vars.r_n: 5,
            pybamm.standard_spatial_vars.r_p: 5,
        }

        sim = pybamm.Simulation(model, var_pts=var_pts)

        t_eval = np.linspace(0, 100, 3)
        x = np.linspace(0, 1, tot_pts)

        for i in np.arange(1, len(t_eval) - 1):
            dt = t_eval[i + 1] - t_eval[i]
            T = (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * 100 * t_eval[i]))[
                :, np.newaxis
            ]
            external_variables = {"Cell temperature": T}
            sim.step(dt, external_variables=external_variables)

    @unittest.skipIf(not pybamm.have_idaklu(), "idaklu solver is not installed")
    def test_dae_external_temperature(self):

        model_options = {"thermal": "x-full", "external submodels": ["thermal"]}

        model = pybamm.lithium_ion.DFN(model_options)

        neg_pts = 5
        sep_pts = 3
        pos_pts = 5
        tot_pts = neg_pts + sep_pts + pos_pts

        var_pts = {
            pybamm.standard_spatial_vars.x_n: neg_pts,
            pybamm.standard_spatial_vars.x_s: sep_pts,
            pybamm.standard_spatial_vars.x_p: pos_pts,
            pybamm.standard_spatial_vars.r_n: 5,
            pybamm.standard_spatial_vars.r_p: 5,
        }

        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(model, var_pts=var_pts, solver=solver)
        sim.build()

        t_eval = np.linspace(0, 100, 3)
        x = np.linspace(0, 1, tot_pts)

        for i in np.arange(1, len(t_eval) - 1):
            dt = t_eval[i + 1] - t_eval[i]
            T = (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * 100 * t_eval[i]))[
                :, np.newaxis
            ]
            external_variables = {"Cell temperature": T}
            sim.step(dt, external_variables=external_variables)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
