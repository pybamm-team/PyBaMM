#
# Tests the Timer class.
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
import os
import pybamm
import unittest


class TestUtil(unittest.TestCase):
    """
    Test the functionality in util.py
    """

    def test_load_function(self):
        # Test filename ends in '.py'
        with self.assertRaisesRegex(
            ValueError, "Expected filename.py, but got doesnotendindotpy"
        ):
            pybamm.load_function("doesnotendindotpy")

        # Test exception if absolute file not found
        with self.assertRaisesRegex(
            ValueError, "is an absolute path, but the file is not found"
        ):
            nonexistent_abs_file = os.path.join(os.getcwd(), "i_dont_exist.py")
            pybamm.load_function(nonexistent_abs_file)

        # Test exception if relative file not found
        with self.assertRaisesRegex(
            ValueError, "cannot be found in the PyBaMM directory"
        ):
            pybamm.load_function("i_dont_exist.py")

        # Test exception if relative file found more than once
        with self.assertRaisesRegex(
            ValueError, "found multiple times in the PyBaMM directory"
        ):
            pybamm.load_function("__init__.py")

        # Test exception if no matching function found in module
        with self.assertRaisesRegex(ValueError, "No function .+ found in module .+"):
            pybamm.load_function("process_symbol_bad_function.py")

        # Test function load with absolute path
        abs_test_path = os.path.join(
            os.getcwd(),
            "tests",
            "unit",
            "test_parameters",
            "data",
            "process_symbol_test_function.py",
        )
        self.assertTrue(os.path.isfile(abs_test_path))
        func = pybamm.load_function(abs_test_path)
        self.assertEqual(func(2), 246)

        # Test function load with relative path
        func = pybamm.load_function("process_symbol_test_function.py")
        self.assertEqual(func(3), 369)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
