#
# Compare basic models with full models
#
import pybamm

import numpy as np
import unittest


class TestCompareBasicModels(unittest.TestCase):
    def test_compare_dfns(self):
        basic_dfn = pybamm.lithium_ion.BasicDFN()
        dfn = pybamm.lithium_ion.DFN()

        # Solve basic DFN
        basic_sim = pybamm.Simulation(basic_dfn)
        t_eval = np.linspace(0, 3600)
        basic_sim.solve(t_eval)
        basic_sol = basic_sim.solution

        # Solve main DFN
        sim = pybamm.Simulation(dfn)
        t_eval = np.linspace(0, 3600)
        sim.solve(t_eval)
        sol = sim.solution

        # Compare solution data
        np.testing.assert_allclose(basic_sol.t, sol.t)
        # Compare variables
        for name in basic_dfn.variables:
            np.testing.assert_allclose(
                basic_sol[name].entries, sol[name].entries, rtol=1e-3
            )

    def test_compare_dfns_composite(self):
        basic_dfn = pybamm.lithium_ion.BasicDFNComposite()
        dfn = pybamm.lithium_ion.DFN(
            {
                "particle phases": ("2", "1"),
                "open-circuit potential": (("single", "current sigmoid"), "single"),
            }
        )
        parameter_values = pybamm.ParameterValues("Chen2020_composite")

        # Solve basic DFN
        basic_sim = pybamm.Simulation(basic_dfn, parameter_values=parameter_values)
        t_eval = np.linspace(0, 3600)
        basic_sim.solve(t_eval)
        basic_sol = basic_sim.solution

        # Solve main DFN
        sim = pybamm.Simulation(dfn, parameter_values=parameter_values)
        t_eval = np.linspace(0, 3600)
        sim.solve(t_eval)
        sol = sim.solution

        # Compare solution data
        np.testing.assert_allclose(basic_sol.t, sol.t)
        # Compare variables
        for name in basic_dfn.variables:
            np.testing.assert_allclose(
                basic_sol[name].entries, sol[name].entries, rtol=1e-2, atol=1e-6
            )

    def test_compare_spms(self):
        basic_spm = pybamm.lithium_ion.BasicSPM()
        spm = pybamm.lithium_ion.SPM()

        # Solve basic SPM
        basic_sim = pybamm.Simulation(basic_spm)
        t_eval = np.linspace(0, 3600)
        basic_sim.solve(t_eval)
        basic_sol = basic_sim.solution

        # Solve main SPM
        sim = pybamm.Simulation(spm)
        t_eval = np.linspace(0, 3600)
        sim.solve(t_eval)
        sol = sim.solution

        # Compare solution data
        np.testing.assert_allclose(basic_sol.t, sol.t)
        # Compare variables
        for name in basic_spm.variables:
            np.testing.assert_allclose(
                basic_sol[name].entries, sol[name].entries, rtol=1e-5
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
