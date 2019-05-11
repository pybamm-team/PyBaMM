#
# Tests for the Reaction diffusion model
#
import pybamm
import tests

import numpy as np
import unittest


class TestReactionDiffusionModel(unittest.TestCase):
    def test_basic_processing(self):
        model = pybamm.ReactionDiffusionModel()

        modeltest = tests.StandardModelTest(model)
        modeltest.test_all()

    def test_optimisations(self):
        model = pybamm.ReactionDiffusionModel()
        optimtest = tests.OptimisationsTest(model)

        original = optimtest.evaluate_model()
        simplified = optimtest.evaluate_model(simplify=True)
        using_known_evals = optimtest.evaluate_model(use_known_evals=True)
        simp_and_known = optimtest.evaluate_model(simplify=True, use_known_evals=True)
        np.testing.assert_array_almost_equal(original, simplified)
        np.testing.assert_array_almost_equal(original, using_known_evals)
        np.testing.assert_array_almost_equal(original, simp_and_known)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
