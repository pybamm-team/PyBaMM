#
# Custom unittest.TestCase class for pybamm
#
import unittest


class TestCase(unittest.TestCase):
    """
    Custom unittest.TestCase class for pybamm
    """

    def assertDomainEqual(self, a, b):
        "Check that two domains are equal, ignoring empty domains"
        a_dict = {k: v for k, v in a.items() if v != []}
        b_dict = {k: v for k, v in b.items() if v != []}
        self.assertEqual(a_dict, b_dict)
