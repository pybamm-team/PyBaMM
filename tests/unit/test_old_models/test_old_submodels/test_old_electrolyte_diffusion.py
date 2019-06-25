#
# Tests for the electrolyte diffusion submodel
#
import pybamm
import unittest


class TestOldElectrolyteDiffusion(unittest.TestCase):
    def test_use_epsilon_param(self):
        parameters = pybamm.standard_parameters_lithium_ion
        model = pybamm.old_electrolyte_diffusion.OldStefanMaxwell(parameters)
        c_e = pybamm.Variable("c")
        variables = {"Electrolyte concentration": c_e}
        model.set_leading_order_system(variables, {})


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
unittest.main()
