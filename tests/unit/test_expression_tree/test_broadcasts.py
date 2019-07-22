#
# Tests for the Broadcast class
#
import pybamm
import numpy as np
import unittest


class TestBroadcasts(unittest.TestCase):
    def test_broadcast(self):
        a = pybamm.Symbol("a")
        broad_a = pybamm.Broadcast(a, ["negative electrode"])
        self.assertEqual(broad_a.name, "broadcast")
        self.assertEqual(broad_a.children[0].name, a.name)
        self.assertEqual(broad_a.domain, ["negative electrode"])

        b = pybamm.Symbol("b", domain=["negative electrode"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.Broadcast(b, ["separator"])

    def test_broadcast_number(self):
        broad_a = pybamm.Broadcast(1, ["negative electrode"])
        self.assertEqual(broad_a.name, "broadcast")
        self.assertIsInstance(broad_a.children[0], pybamm.Symbol)
        self.assertEqual(broad_a.children[0].evaluate(), np.array([1]))
        self.assertEqual(broad_a.domain, ["negative electrode"])

        b = pybamm.Symbol("b", domain=["negative electrode"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.Broadcast(b, ["separator"])

    def test_broadcast_type(self):
        import ipdb

        ipdb.set_trace()


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
