#
# Tests the logger class.
#
from tests import TestCase
import pybamm
import unittest


class TestLogger(TestCase):
    def test_logger(self):
        logger = pybamm.logger
        self.assertEqual(logger.level, 30)
        pybamm.set_logging_level("INFO")
        self.assertEqual(logger.level, 20)
        pybamm.set_logging_level("ERROR")
        self.assertEqual(logger.level, 40)
        pybamm.set_logging_level("VERBOSE")
        self.assertEqual(logger.level, 15)
        pybamm.set_logging_level("NOTICE")
        self.assertEqual(logger.level, 25)
        pybamm.set_logging_level("SUCCESS")
        self.assertEqual(logger.level, 35)

        pybamm.set_logging_level("SPAM")
        self.assertEqual(logger.level, 5)
        pybamm.logger.spam("Test spam level")
        pybamm.logger.verbose("Test verbose level")
        pybamm.logger.notice("Test notice level")
        pybamm.logger.success("Test success level")

        # reset
        pybamm.set_logging_level("WARNING")

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            pybamm.get_new_logger("test", None)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
