#
# Tests for the Broadcast class
#
import pybamm
import numpy as np
import unittest


class TestBroadcasts(unittest.TestCase):
    def test_primary_broadcast(self):
        a = pybamm.Symbol("a")
        broad_a = pybamm.PrimaryBroadcast(a, ["negative electrode"])
        self.assertEqual(broad_a.name, "broadcast")
        self.assertEqual(broad_a.children[0].name, a.name)
        self.assertEqual(broad_a.domain, ["negative electrode"])

        a = pybamm.Symbol(
            "a",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        broad_a = pybamm.PrimaryBroadcast(a, ["negative particle"])
        self.assertEqual(broad_a.domain, ["negative particle"])
        self.assertEqual(
            broad_a.auxiliary_domains,
            {"secondary": ["negative electrode"], "tertiary": ["current collector"]},
        )

        a = pybamm.Symbol("a", domain="current collector")
        with self.assertRaisesRegex(
            pybamm.DomainError, "Primary broadcast from current collector"
        ):
            pybamm.PrimaryBroadcast(a, "bad domain")
        a = pybamm.Symbol("a", domain="negative electrode")
        with self.assertRaisesRegex(
            pybamm.DomainError, "Primary broadcast from electrode"
        ):
            pybamm.PrimaryBroadcast(a, "current collector")
        a = pybamm.Symbol("a", domain="negative particle")
        with self.assertRaisesRegex(
            pybamm.DomainError, "Cannot do primary broadcast from particle domain"
        ):
            pybamm.PrimaryBroadcast(a, "current collector")

    def test_secondary_broadcast(self):
        a = pybamm.Symbol(
            "a",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "current collector"},
        )
        broad_a = pybamm.SecondaryBroadcast(a, ["negative electrode"])
        self.assertEqual(broad_a.domain, ["negative particle"])
        self.assertEqual(
            broad_a.auxiliary_domains,
            {"secondary": ["negative electrode"], "tertiary": ["current collector"]},
        )

        a = pybamm.Symbol("a", domain="negative particle")
        with self.assertRaisesRegex(
            pybamm.DomainError, "Secondary broadcast from particle"
        ):
            pybamm.SecondaryBroadcast(a, "current collector")
        a = pybamm.Symbol("a", domain="negative electrode")
        with self.assertRaisesRegex(
            pybamm.DomainError, "Secondary broadcast from electrode"
        ):
            pybamm.SecondaryBroadcast(a, "negative particle")

        a = pybamm.Symbol("a", domain="current collector")
        with self.assertRaisesRegex(
            pybamm.DomainError, "Cannot do secondary broadcast"
        ):
            pybamm.SecondaryBroadcast(a, "electrode")

    def test_full_broadcast(self):
        a = pybamm.Symbol("a")
        broad_a = pybamm.FullBroadcast(a, ["negative electrode"], "current collector")
        self.assertEqual(broad_a.domain, ["negative electrode"])
        self.assertEqual(broad_a.auxiliary_domains["secondary"], ["current collector"])

    def test_full_broadcast_number(self):
        broad_a = pybamm.FullBroadcast(1, ["negative electrode"], None)
        self.assertEqual(broad_a.name, "broadcast")
        self.assertIsInstance(broad_a.children[0], pybamm.Symbol)
        self.assertEqual(broad_a.children[0].evaluate(), np.array([1]))
        self.assertEqual(broad_a.domain, ["negative electrode"])

        a = pybamm.Symbol("a", domain="current collector")
        with self.assertRaisesRegex(pybamm.DomainError, "Cannot do full broadcast"):
            pybamm.FullBroadcast(a, "electrode", None)

    def test_ones_like(self):
        a = pybamm.Variable("a")
        ones_like_a = pybamm.ones_like(a)
        self.assertEqual(ones_like_a.id, pybamm.Scalar(1).id)

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

        b = pybamm.Variable("b", domain="current collector")
        ones_like_ab = pybamm.ones_like(b, a)
        self.assertIsInstance(ones_like_ab, pybamm.FullBroadcast)
        self.assertEqual(ones_like_ab.name, "broadcast")
        self.assertEqual(ones_like_ab.domain, a.domain)
        self.assertEqual(ones_like_ab.auxiliary_domains, a.auxiliary_domains)

    def test_broadcast_to_edges(self):
        a = pybamm.Symbol("a")
        broad_a = pybamm.PrimaryBroadcastToEdges(a, ["negative electrode"])
        self.assertEqual(broad_a.name, "broadcast to edges")
        self.assertEqual(broad_a.children[0].name, a.name)
        self.assertEqual(broad_a.domain, ["negative electrode"])
        self.assertTrue(broad_a.evaluates_on_edges("primary"))

        a = pybamm.Symbol(
            "a",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "current collector"},
        )
        broad_a = pybamm.SecondaryBroadcastToEdges(a, ["negative electrode"])
        self.assertEqual(broad_a.domain, ["negative particle"])
        self.assertEqual(
            broad_a.auxiliary_domains,
            {"secondary": ["negative electrode"], "tertiary": ["current collector"]},
        )
        self.assertTrue(broad_a.evaluates_on_edges("primary"))

        a = pybamm.Symbol("a")
        broad_a = pybamm.FullBroadcastToEdges(
            a, ["negative electrode"], "current collector"
        )
        self.assertEqual(broad_a.domain, ["negative electrode"])
        self.assertEqual(broad_a.auxiliary_domains["secondary"], ["current collector"])
        self.assertTrue(broad_a.evaluates_on_edges("primary"))


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
