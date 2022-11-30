#
# Test edge cases for initial SOC
#
import pybamm
import unittest


class TestInitialSOC(unittest.TestCase):
    def test_interpolant_parameter_sets(self):
        model = pybamm.lithium_ion.SPM()
        for param in ["OKane2022", "Ai2020"]:
            with self.subTest(param=param):
                parameter_values = pybamm.ParameterValues(param)
                sim = pybamm.Simulation(model=model, parameter_values=parameter_values)
                sim.solve([0, 3600], initial_soc=0.2)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
