#
# Tests for the Base Solver class
#
import pybamm
import numpy as np

import unittest


class TestBaseSolver(unittest.TestCase):
    def test_base_solver_init(self):
        solver = pybamm.BaseSolver(rtol=1e-2, atol=1e-4)
        self.assertEqual(solver.rtol, 1e-2)
        self.assertEqual(solver.atol, 1e-4)

        solver.rtol = 1e-5
        self.assertEqual(solver.rtol, 1e-5)
        solver.rtol = 1e-7
        self.assertEqual(solver.rtol, 1e-7)

        with self.assertRaises(NotImplementedError):
            solver.compute_solution(None, None)
        with self.assertRaises(NotImplementedError):
            solver.set_up(None)

    def test_step_or_solve_empty_model(self):
        model = pybamm.BaseModel()
        solver = pybamm.BaseSolver()
        with self.assertRaisesRegex(pybamm.ModelError, "Cannot step empty model"):
            solver.step(model, None)
        with self.assertRaisesRegex(pybamm.ModelError, "Cannot solve empty model"):
            solver.solve(model, None)

    def test_set_external_variables(self):
        options = {"thermal": "x-full", "external submodels": ["thermal"]}
        model = pybamm.lithium_ion.SPM(options)
        sim = pybamm.Simulation(model)
        sim.build()
        solver = pybamm.BaseSolver()

        T = np.ones((60, 1))
        external_variables = {"Cell temperature": T}
        solver.set_external_variables(sim.built_model, external_variables)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
