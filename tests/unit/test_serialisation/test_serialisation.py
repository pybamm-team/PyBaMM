#
# Tests for the serialisation class
#
from tests import TestCase
import tests
import pybamm

pybamm.settings.debug_mode = True

import numpy as np
import unittest


class TestSerialiseModels(TestCase):
    # test lithium models
    def test_spm_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lithium_ion.SPM()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_equal(solution.all_ys[x], new_solution.all_ys[x])

    def test_spme_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lithium_ion.SPMe()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_equal(solution.all_ys[x], new_solution.all_ys[x])

    def test_mpm_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lithium_ion.MPM()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x]
            )

    def test_dfn_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lithium_ion.DFN()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x]
            )

    def test_newman_tobias_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lithium_ion.NewmanTobias()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x]
            )

    def test_msmr_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x], decimal=3
            )

    # test lead-acid models
    def test_lead_acid_full_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lead_acid.Full()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x]
            )

    def test_loqs_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.lead_acid.LOQS()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x]
            )

    def test_thevenin_serialisation_recreation(self):
        t = [0, 3600]

        model = pybamm.equivalent_circuit.Thevenin()
        sim = pybamm.Simulation(model)
        solution = sim.solve(t)

        sim.save_model("test_model")

        new_model = pybamm.load_model("test_model.json")
        new_solver = new_model.default_solver
        new_solution = new_solver.solve(new_model, t)

        for x, val in enumerate(solution.all_ys):
            np.testing.assert_array_almost_equal(
                solution.all_ys[x], new_solution.all_ys[x]
            )


class TestSerialiseExpressionTree(TestCase):
    def test_tree_walk(self):
        pass


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
