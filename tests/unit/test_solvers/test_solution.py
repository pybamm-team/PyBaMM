#
# Tests for the Solution class
#
import pybamm
import unittest
import numpy as np


class TestSolution(unittest.TestCase):
    def test_init(self):
        t = np.linspace(0, 1)
        y = np.tile(t, (20, 1))
        sol = pybamm.Solution(t, y)
        np.testing.assert_array_equal(sol.t, t)
        np.testing.assert_array_equal(sol.y, y)
        self.assertEqual(sol.t_event, None)
        self.assertEqual(sol.y_event, None)
        self.assertEqual(sol.termination, "final time")
        self.assertEqual(sol.inputs, {})
        self.assertIsInstance(sol.model, pybamm.BaseModel)

    def test_append(self):
        # Set up first solution
        t1 = np.linspace(0, 1)
        y1 = np.tile(t1, (20, 1))
        sol1 = pybamm.Solution(t1, y1)
        sol1.solve_time = 1.5

        # Set up second solution
        t2 = np.linspace(1, 2)
        y2 = np.tile(t2, (20, 1))
        sol2 = pybamm.Solution(t2, y2)
        sol2.solve_time = 1
        sol1.append(sol2)

        # Test
        self.assertEqual(sol1.solve_time, 2.5)
        np.testing.assert_array_equal(sol1.t, np.concatenate([t1, t2[1:]]))
        np.testing.assert_array_equal(sol1.y, np.concatenate([y1, y2[:, 1:]], axis=1))

    def test_total_time(self):
        sol = pybamm.Solution([], None)
        sol.set_up_time = 0.5
        sol.solve_time = 1.2
        self.assertEqual(sol.total_time, 1.7)

    def test_getitem(self):
        # test create a new processed variable
        # test call an already created variable
        pass

    def test_save(self):
        # test save
        # test save data
        pass


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
