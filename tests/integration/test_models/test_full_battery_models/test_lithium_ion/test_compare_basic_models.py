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
        basic_dfn_eqns = {**basic_dfn.rhs, **basic_dfn.algebraic}
        dfn_eqns = {**dfn.rhs, **dfn.algebraic}
        self.assertEqual(
            [var.name for var in dfn_eqns.keys()],
            [var.name for var in basic_dfn_eqns.keys()],
        )

        # Solve basic DFN
        basic_sim = pybamm.Simulation(basic_dfn)
        t_eval = np.linspace(0, 1)
        basic_sim.solve(t_eval)
        basic_sol = basic_sim.solution

        # Solve main DFN
        sim = pybamm.Simulation(dfn)
        t_eval = np.linspace(0, 1)
        sim.solve(t_eval)
        sol = sim.solution

        # Compare solution data
        np.testing.assert_array_almost_equal(basic_sol.y, sol.y, decimal=4)
        np.testing.assert_array_almost_equal(basic_sol.t, sol.t, decimal=4)
        # Compare variables
        for name in basic_dfn.variables:
            np.testing.assert_array_almost_equal(
                basic_sol[name].entries, sol[name].entries, decimal=4
            )

    def test_compare_spms(self):
        basic_spm = pybamm.lithium_ion.BasicSPM()
        spm = pybamm.lithium_ion.SPM()
        basic_spm_eqns = {**basic_spm.rhs, **basic_spm.algebraic}
        spm_eqns = {**spm.rhs, **spm.algebraic}
        self.assertEqual(
            [var.name for var in spm_eqns.keys()],
            [var.name for var in basic_spm_eqns.keys()],
        )

        # Solve basic SPM
        basic_sim = pybamm.Simulation(basic_spm)
        t_eval = np.linspace(0, 1)
        basic_sim.solve(t_eval)
        basic_sol = basic_sim.solution

        # Solve main SPM
        sim = pybamm.Simulation(spm)
        t_eval = np.linspace(0, 1)
        sim.solve(t_eval)
        sol = sim.solution

        # Compare solution data
        np.testing.assert_array_almost_equal(basic_sol.y, sol.y)
        np.testing.assert_array_almost_equal(basic_sol.t, sol.t)
        # Compare variables
        for name in basic_spm.variables:
            np.testing.assert_array_almost_equal(
                basic_sol[name].entries, sol[name].entries
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
