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

    def test_broadcast_number(self):
        broad_a = pybamm.Broadcast(1, ["negative electrode"])
        self.assertEqual(broad_a.name, "broadcast")
        self.assertIsInstance(broad_a.children[0], pybamm.Symbol)
        self.assertEqual(broad_a.children[0].evaluate(), np.array([1]))
        self.assertEqual(broad_a.domain, ["negative electrode"])

    def test_broadcast_type(self):
        a = pybamm.Symbol("a", domain="current collector")
        with self.assertRaisesRegex(ValueError, "Variables on the current collector"):
            pybamm.Broadcast(a, "electrode")

    def test_ones_like(self):
        a = pybamm.Variable(
            "a",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        ones_like_a = pybamm.ones_like(a)
        self.assertIsInstance(ones_like_a, pybamm.FullBroadcast)
        self.assertEqual(ones_like_a.name, "broadcast")
        self.assertEqual(ones_like_a.domain, a.domain)
        self.assertEqual(ones_like_a.auxiliary_domains, a.auxiliary_domains)

        b = pybamm.Variable("b", domain="negative electrode")
        ones_like_ab = pybamm.ones_like(b, a)
        self.assertIsInstance(ones_like_ab, pybamm.FullBroadcast)
        self.assertEqual(ones_like_ab.name, "broadcast")
        self.assertEqual(ones_like_ab.domain, a.domain)
        self.assertEqual(ones_like_ab.auxiliary_domains, a.auxiliary_domains)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
