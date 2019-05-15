#
# Tests for the Reaction diffusion model
#
import pybamm
import unittest


class TestReactionDiffusionModel(unittest.TestCase):
    def test_well_posed(self):
        model = pybamm.ReactionDiffusionModel()
        model.check_well_posedness()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
