#
# Tests the logger class.
#
import pybamm
import unittest


class TestLogger(unittest.TestCase):
    def test_logger(self):
        logger = pybamm.logger
        self.assertEqual(logger.level, 0)
        pybamm.set_logging_level("INFO")
        self.assertEqual(logger.level, 20)
        pybamm.set_logging_level("WARNING")
        self.assertEqual(logger.level, 30)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
