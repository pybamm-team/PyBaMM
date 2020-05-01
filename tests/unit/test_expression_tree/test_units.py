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

        # empty units
        no_units = pybamm.Units(None)
        self.assertEqual(no_units.units, {})
        self.assertEqual(no_units.units_str, "[-]")

        # errors
        with self.assertRaisesRegex(pybamm.UnitsError, "not recognized"):
            pybamm.Units("[notaunit]")
        with self.assertRaisesRegex(pybamm.UnitsError, "Units should start with"):
            pybamm.Units("m.s-1")

        # Non-standard units
        # [W]
        # [S]
        # [F]
        # volts = pybamm.Units("[V]")

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

        speed_times_conc = speed * conc
        self.assertEqual(speed_times_conc.units, {"m": -2, "mol": 1, "s": -1})

        speed_over_conc = speed / conc
        self.assertEqual(speed_over_conc.units, {"m": 4, "mol": -1, "s": -1})

        conc_over_speed = conc / speed
        self.assertEqual(conc_over_speed.units, {"m": -4, "mol": 1, "s": 1})

        speed_cubed = speed ** 3.5
        self.assertEqual(speed_cubed.units, {"m": 3.5, "s": -3.5})

    def test_symbol_units(self):
        a = pybamm.Symbol("a")
        c = pybamm.Symbol("c", units="[mol.m-3]")
        v = pybamm.Symbol("v", units="[m.s-1]")

        self.assertIsInstance(a._units_class, pybamm.Units)
        self.assertEqual(a.units, "[-]")

        self.assertIsInstance(c._units_class, pybamm.Units)
        self.assertEqual(c.units, "[mol.m-3]")

        self.assertEqual((c + c).units, "[mol.m-3]")
        self.assertEqual((c - c).units, "[mol.m-3]")
        self.assertEqual((c * v).units, "[mol.m-2.s-1]")
        self.assertEqual((c / v).units, "[mol.s.m-4]")
        self.assertEqual((v / c).units, "[m4.mol-1.s-1]")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
