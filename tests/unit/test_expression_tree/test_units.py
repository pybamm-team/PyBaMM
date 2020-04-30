#
# Test for the Units class
#
import pybamm
import unittest


class TestUnits(unittest.TestCase):
    def test_unit_init(self):
        speed_str = pybamm.Units("[m.s-1]")
        speed_dict = pybamm.Units({"m": 1, "s": -1})
        self.assertEqual(speed_str.units, {"m": 1, "s": -1})
        self.assertEqual(speed_dict.units_str, "[m.s-1]")

        with self.assertRaisesRegex(pybamm.UnitsError, "not recognized"):
            pybamm.Units("[notaunit]")
        with self.assertRaisesRegex(pybamm.UnitsError, "Units should start with"):
            pybamm.Units("m.s-1")

        # Non-standard units
        volts = pybamm.Units("[V]")

    def test_units_operations(self):
        speed = pybamm.Units("[m.s-1]")
        conc = pybamm.Units("[mol.m-3]")

        speed_sum = speed + speed
        self.assertEqual(speed_sum.units_str, "[m.s-1]")
        with self.assertRaisesRegex(pybamm.UnitsError, "Cannot add"):
            speed + conc

        speed_diff = speed - speed
        self.assertEqual(speed_diff.units_str, "[m.s-1]")
        with self.assertRaisesRegex(pybamm.UnitsError, "Cannot subtract"):
            speed - conc

        # speed_times_conc = speed * conc
        # self.assertEqual(speed_times_conc.units, {"m": -2, "mol": 1, "s": -1})


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
