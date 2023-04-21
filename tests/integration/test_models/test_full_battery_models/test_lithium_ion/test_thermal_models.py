#
# Tests for the thermal lithium-ion models produce consistent
# thermal response
#
import pybamm
import numpy as np
import unittest
from tests import TestCase


class TestThermal(TestCase):
    def test_consistent_cooling(self):
        # use spme for comparison instead of spm as
        # much larger realistic temperature rises
        # so that errors can be more easily observed
        C_rate = 5
        options = {"thermal": "x-lumped"}
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
        solutions = {}

        for model_name, model in models.items():
            var_pts = {"x_n": 3, "x_s": 3, "x_p": 3, "r_n": 3, "r_p": 3, "y": 5, "z": 5}
            parameter_values = pybamm.ParameterValues("NCA_Kim2011")

            # high thermal and electrical conductivity in current collectors
            parameter_values.update(
                {
                    "Negative current collector"
                    + " surface heat transfer coefficient [W.m-2.K-1]": 10,
                    "Positive current collector"
                    + " surface heat transfer coefficient [W.m-2.K-1]": 5,
                    "Negative tab heat transfer coefficient [W.m-2.K-1]": 250,
                    "Positive tab heat transfer coefficient [W.m-2.K-1]": 250,
                    "Edge heat transfer coefficient [W.m-2.K-1]": 100,
                    "Negative current collector"
                    + " thermal conductivity [W.m-1.K-1]": 267.467 * 100000,
                    "Positive current collector"
                    + " thermal conductivity [W.m-1.K-1]": 158.079 * 100000,
                    "Negative current collector conductivity [S.m-1]": 1e10,
                    "Positive current collector conductivity [S.m-1]": 1e10,
                }
            )

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

        self.assertGreater(1e-5, err(solutions["SPMe 1D"], solutions["SPMe 1+1D"]))
        self.assertGreater(1e-5, err(solutions["SPMe 1D"], solutions["SPMe 2+1D"]))
        self.assertGreater(1e-5, err(solutions["SPMe 1+1D"], solutions["SPMe 2+1D"]))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
