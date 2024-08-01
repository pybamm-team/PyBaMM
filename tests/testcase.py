#
# Custom TestCase class for pybamm
#
import unittest


class TestCase(unittest.TestCase):
    """
    Custom TestCase class for PyBaMM
    TO BE REMOVED
    """

    def assertDomainEqual(self, a, b):
        "Check that two domains are equal, ignoring empty domains"
        a_dict = {k: v for k, v in a.items() if v != []}
        b_dict = {k: v for k, v in b.items() if v != []}
        self.assertEqual(a_dict, b_dict)
