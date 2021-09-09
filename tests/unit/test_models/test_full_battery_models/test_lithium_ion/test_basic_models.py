#
# Tests for the basic lithium-ion models
#
import pybamm
import unittest


class TestBasicModels(unittest.TestCase):
    def test_dfn_well_posed(self):
        model = pybamm.lithium_ion.BasicDFN()
        model.check_well_posedness()

        copy = model.new_copy()
        copy.check_well_posedness()

    def test_spm_well_posed(self):
        model = pybamm.lithium_ion.BasicSPM()
        model.check_well_posedness()

        copy = model.new_copy()
        copy.check_well_posedness()

    def test_dfn_half_cell_well_posed(self):
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)
        model.check_well_posedness()

        copy = model.new_copy()
        copy.check_well_posedness()

        options = {"working electrode": "negative"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)
        model.check_well_posedness()

        copy = model.new_copy()
        copy.check_well_posedness()

    def test_dfn_half_cell_simulation_with_experiment_error(self):
        options = {"working electrode": "negative"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)
        experiment = pybamm.Experiment(
            [("Discharge at C/10 for 10 hours or until 3.5 V")]
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "BasicDFNHalfCell is not compatible with experiment simulations yet.",
        ):
            pybamm.Simulation(model, experiment=experiment)

    def test_basic_dfn_half_cell_simulation(self):
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"working electrode": "positive"}
        )
        chemistry = pybamm.parameter_sets.Chen2020_plating
        param = pybamm.ParameterValues(chemistry=chemistry)
        param["Current function [A]"] = 2.5
        sim = pybamm.Simulation(model=model, parameter_values=param)
        sim.solve([0, 100])
        self.assertTrue(isinstance(sim.solution, pybamm.solvers.solution.Solution))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
