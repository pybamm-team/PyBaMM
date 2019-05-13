import pybamm
import unittest


class TestQuickPlot(unittest.TestCase):
    """
    Tests that QuickPlot is created correctly
    """

    def test_failure(self):
        with self.assertRaisesRegex(TypeError, "'models' must be"):
            pybamm.QuickPlot(1, None, None)
        with self.assertRaisesRegex(TypeError, "'solvers' must be"):
            pybamm.QuickPlot(pybamm.BaseModel(), None, 1)
        with self.assertRaisesRegex(ValueError, "must provide the same"):
            pybamm.QuickPlot(
                pybamm.BaseModel(), None, [pybamm.BaseSolver(), pybamm.BaseSolver()]
            )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
