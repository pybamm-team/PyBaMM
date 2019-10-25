#
# Tests the Timer class.
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
import pybamm
import unittest


class TestTimer(unittest.TestCase):
    """
    Tests the basic methods of the Timer class.
    """

    def __init__(self, name):
        super(TestTimer, self).__init__(name)

    def test_timing(self):
        t = pybamm.Timer()
        a = t.time()
        self.assertGreaterEqual(a, 0)
        for _ in range(100):
            self.assertGreater(t.time(), a)
        a = t.time()
        t.reset()
        b = t.time()
        self.assertGreaterEqual(b, 0)
        self.assertLess(b, a)

    def test_timer_format(self):
        import sys

        t = pybamm.Timer()
        self.assertEqual(t.format(1e-3), "0.001 seconds")
        self.assertEqual(t.format(0.000123456789), "0.000123456789 seconds")
        self.assertEqual(t.format(0.123456789), "0.12 seconds")
        if sys.hexversion < 0x3000000:
            self.assertEqual(t.format(2), "2.0 seconds")
        else:
            self.assertEqual(t.format(2), "2 seconds")
        self.assertEqual(t.format(2.5), "2.5 seconds")
        self.assertEqual(t.format(12.5), "12.5 seconds")
        self.assertEqual(t.format(59.41), "59.41 seconds")
        self.assertEqual(t.format(59.4126347547), "59.41 seconds")
        self.assertEqual(t.format(60.2), "1 minute, 0 seconds")
        self.assertEqual(t.format(61), "1 minute, 1 second")
        self.assertEqual(t.format(121), "2 minutes, 1 second")
        self.assertEqual(
            t.format(604800), "1 week, 0 days, 0 hours, 0 minutes, 0 seconds"
        )
        self.assertEqual(
            t.format(2 * 604800 + 3 * 3600 + 60 + 4),
            "2 weeks, 0 days, 3 hours, 1 minute, 4 seconds",
        )

        # Test without argument
        self.assertIsInstance(t.format(), str)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
