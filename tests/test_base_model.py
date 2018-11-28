#
# Tests for the base model class
#
import pybamm

import unittest


class TestBaseModel(unittest.TestCase):
    def test_base_model_basic(self):
        model = pybamm.BaseModel()
        # __str__
        self.assertEqual(str(model), "Base Model")

        # domains
        model.variables = [("c", "xc"), ("en", "xcn"), ("epsn", "xcn")]
        self.assertEqual(model.domains(), set(["xc", "xcn"]))

        # set simulation
        model.set_simulation(1, 2, 3)
        self.assertEqual(model.param, 1)
        self.assertEqual(model.operators, 2)
        self.assertEqual(model.mesh, 3)

        # NotImplementedErrors
        with self.assertRaises(NotImplementedError):
            model.submodels


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
