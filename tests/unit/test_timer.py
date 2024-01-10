#
# Tests the Timer class.
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
import pybamm
import unittest
from tests import TestCase


class TestTimer(TestCase):
    """
    Tests the basic methods of the Timer class.
    """

    def __init__(self, name):
        super().__init__(name)

    def test_timing(self):
        t = pybamm.Timer()
        a = t.time().value
        self.assertGreaterEqual(a, 0)
        for _ in range(100):
            self.assertGreater(t.time().value, a)
        a = t.time().value
        t.reset()
        b = t.time().value
        self.assertGreaterEqual(b, 0)
        self.assertLess(b, a)

    def test_timer_format(self):
        self.assertEqual(str(pybamm.TimerTime(1e-9)), "1.000 ns")
        self.assertEqual(str(pybamm.TimerTime(0.000000123456789)), "123.457 ns")
        self.assertEqual(str(pybamm.TimerTime(1e-6)), "1.000 us")
        self.assertEqual(str(pybamm.TimerTime(0.000123456789)), "123.457 us")
        self.assertEqual(str(pybamm.TimerTime(0.999e-3)), "999.000 us")
        self.assertEqual(str(pybamm.TimerTime(1e-3)), "1.000 ms")
        self.assertEqual(str(pybamm.TimerTime(0.123456789)), "123.457 ms")
        self.assertEqual(str(pybamm.TimerTime(2)), "2.000 s")
        self.assertEqual(str(pybamm.TimerTime(2.5)), "2.500 s")
        self.assertEqual(str(pybamm.TimerTime(12.5)), "12.500 s")
        self.assertEqual(str(pybamm.TimerTime(59.41)), "59.410 s")
        self.assertEqual(str(pybamm.TimerTime(59.4126347547)), "59.413 s")
        self.assertEqual(str(pybamm.TimerTime(60.2)), "1 minute, 0 seconds")
        self.assertEqual(str(pybamm.TimerTime(61)), "1 minute, 1 second")
        self.assertEqual(str(pybamm.TimerTime(121)), "2 minutes, 1 second")
        self.assertEqual(
            str(pybamm.TimerTime(604800)),
            "1 week, 0 days, 0 hours, 0 minutes, 0 seconds",
        )
        self.assertEqual(
            str(pybamm.TimerTime(2 * 604800 + 3 * 3600 + 60 + 4)),
            "2 weeks, 0 days, 3 hours, 1 minute, 4 seconds",
        )

        self.assertEqual(repr(pybamm.TimerTime(1.5)), "pybamm.TimerTime(1.5)")

    def test_timer_operations(self):
        self.assertEqual((pybamm.TimerTime(1) + 2).value, 3)
        self.assertEqual((1 + pybamm.TimerTime(1)).value, 2)
        self.assertEqual((pybamm.TimerTime(1) - 2).value, -1)
        self.assertEqual((pybamm.TimerTime(1) - pybamm.TimerTime(2)).value, -1)
        self.assertEqual((1 - pybamm.TimerTime(1)).value, 0)
        self.assertEqual((pybamm.TimerTime(4) * 2).value, 8)
        self.assertEqual((pybamm.TimerTime(4) * pybamm.TimerTime(2)).value, 8)
        self.assertEqual((2 * pybamm.TimerTime(5)).value, 10)
        self.assertEqual((pybamm.TimerTime(4) / 2).value, 2)
        self.assertEqual((pybamm.TimerTime(4) / pybamm.TimerTime(2)).value, 2)
        self.assertEqual((2 / pybamm.TimerTime(5)).value, 2 / 5)

        self.assertTrue(pybamm.TimerTime(1) == pybamm.TimerTime(1))
        self.assertTrue(pybamm.TimerTime(1) != pybamm.TimerTime(2))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
