#
# Tests the Timer class.
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
import pybamm
import unittest


class TestUtil(unittest.TestCase):
    """
    Test the functionality in util.py
    """

    def test_load_function(self):

        # Currently assumes it will receive a filename ending in '.py'
        with self.assertRaisesRegex(ValueError,
                                    'Expected filename.py, but got doesnotendindotpy'):
            pybamm.load_function('doesnotendindotpy')


        # self.assertEqual(my_func(-3), 3)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
