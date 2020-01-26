#
# Test setting up a simulation with an experiment
#
import pybamm
import unittest


class TestSimulationExperiment(unittest.TestCase):
    def test_set_up(self):
        experiment = pybamm.Experiment(
            ["Charge at 1 C for 0.05 hours", "Hold at 4.1 V for 1 hours"],
            {}
            # {
            #     "Reference temperature [K]": 298.15,
            #     "Heat transfer coefficient [W.m-2.K-1]": 10,
            #     "Number of electrodes connected in parallel to make a cell": 1,
            #     "Number of cells connected in series to make a battery": 1,
            #     "Lower voltage cut-off [V]": 3.105,
            #     "Upper voltage cut-off [V]": 4.7,
            #     "C-rate": 1,
            #     "Initial concentration in negative electrode [mol.m-3]": 19986,
            #     "Initial concentration in positive electrode [mol.m-3]": 30730,
            #     "Initial concentration in electrolyte [mol.m-3]": 1000,
            #     "Initial temperature [K]": 298.15,
            # },
        )
        print(experiment.operating_conditions)
        model = pybamm.lithium_ion.DFN()
        sim = pybamm.Simulation(
            model, experiment=experiment, solver=pybamm.CasadiSolver()
        )
        sim.solve()
        sim.plot()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
