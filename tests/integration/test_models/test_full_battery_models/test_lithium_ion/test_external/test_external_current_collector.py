#
# Tests for external submodels
#
import pybamm
import unittest
import numpy as np


class TestExternalCC(unittest.TestCase):
    @unittest.skipIf(not pybamm.have_idaklu(), "idaklu solver is not installed")
    def test_2p1d(self):
        model_options = {
            "current collector": "potential pair",
            "dimensionality": 2,
            "external submodels": ["current collector"],
        }
        model = pybamm.lithium_ion.DFN(model_options)
        yz_pts = 3
        var_pts = {
            pybamm.standard_spatial_vars.x_n: 4,
            pybamm.standard_spatial_vars.x_s: 4,
            pybamm.standard_spatial_vars.x_p: 4,
            pybamm.standard_spatial_vars.r_n: 4,
            pybamm.standard_spatial_vars.r_p: 4,
            pybamm.standard_spatial_vars.y: yz_pts,
            pybamm.standard_spatial_vars.z: yz_pts,
        }
        solver = pybamm.IDAKLUSolver()
        sim = pybamm.Simulation(model, var_pts=var_pts, solver=solver)

        t_eval = np.linspace(0, 0.08, 3)

        for i in np.arange(1, len(t_eval) - 1):
            dt = t_eval[i + 1] - t_eval[i]
            print(t_eval[i])

            # provide phi_s_n and i_cc
            phi_s_n = np.zeros((yz_pts ** 2, 1))
            i_boundary_cc = np.ones((yz_pts ** 2, 1))
            external_variables = {
                "Negative current collector potential": phi_s_n,
                "Current collector current density": i_boundary_cc,
            }

            sim.step(dt, external_variables=external_variables)

            # obtain phi_s_n from the pybamm solution at the current time
            phi_s_p = sim.get_variable_array("Positive current collector potential")

        self.assertTrue(phi_s_p.shape, (yz_pts ** 2, 1))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
