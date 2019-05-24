#
# Tests for the Reaction diffusion model
#
import pybamm
import unittest


class TestReactionDiffusionModel(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.ReactionDiffusionModel()
        model.check_well_posedness()

    def test_default_parameter_values(self):
        model = pybamm.ReactionDiffusionModel()
        self.assertTrue(
            isinstance(model.default_parameter_values, pybamm.ParameterValues)
        )
        self.assertTrue(
            "Partial molar volume of cations [m3.mol-1]"
            in model.default_parameter_values
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
