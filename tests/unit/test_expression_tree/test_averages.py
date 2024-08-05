#
# Tests for the Unary Operator classes
#
import unittest
import numpy as np
import pybamm
from tests import assert_domain_equal


class TestUnaryOperators(unittest.TestCase):
    def test_x_average(self):
        a = pybamm.Scalar(4)
        average_a = pybamm.x_average(a)
        self.assertEqual(average_a, a)

        # average of a broadcast is the child
        average_broad_a = pybamm.x_average(
            pybamm.PrimaryBroadcast(a, ["negative electrode"])
        )
        self.assertEqual(average_broad_a, pybamm.Scalar(4))

        # average of a number times a broadcast is the number times the child
        average_two_broad_a = pybamm.x_average(
            2 * pybamm.PrimaryBroadcast(a, ["negative electrode"])
        )
        self.assertEqual(average_two_broad_a, pybamm.Scalar(8))
        average_t_broad_a = pybamm.x_average(
            pybamm.t * pybamm.PrimaryBroadcast(a, ["negative electrode"])
        )
        self.assertEqual(average_t_broad_a, (pybamm.t * pybamm.Scalar(4)))

        # full broadcasts
        average_broad_a = pybamm.x_average(
            pybamm.FullBroadcast(
                a,
                ["negative particle"],
                {
                    "secondary": "negative particle size",
                    "tertiary": "negative electrode",
                    "quaternary": "current collector",
                },
            )
        )
        average_broad_a_simp = pybamm.FullBroadcast(
            a,
            ["negative particle"],
            {"secondary": "negative particle size", "tertiary": "current collector"},
        )
        self.assertEqual(average_broad_a, average_broad_a_simp)

        # x-average of concatenation of broadcasts
        conc_broad = pybamm.concatenation(
            pybamm.PrimaryBroadcast(1, ["negative electrode"]),
            pybamm.PrimaryBroadcast(2, ["separator"]),
            pybamm.PrimaryBroadcast(3, ["positive electrode"]),
        )
        average_conc_broad = pybamm.x_average(conc_broad)
        self.assertIsInstance(average_conc_broad, pybamm.Division)
        self.assertEqual(average_conc_broad.domain, [])
        # separator and positive electrode only (half-cell model)
        conc_broad = pybamm.concatenation(
            pybamm.PrimaryBroadcast(2, ["separator"]),
            pybamm.PrimaryBroadcast(3, ["positive electrode"]),
        )
        average_conc_broad = pybamm.x_average(conc_broad)
        self.assertIsInstance(average_conc_broad, pybamm.Division)
        self.assertEqual(average_conc_broad.domain, [])
        # with auxiliary domains
        conc_broad = pybamm.concatenation(
            pybamm.FullBroadcast(
                1,
                ["negative electrode"],
                auxiliary_domains={"secondary": "current collector"},
            ),
            pybamm.FullBroadcast(
                2, ["separator"], auxiliary_domains={"secondary": "current collector"}
            ),
            pybamm.FullBroadcast(
                3,
                ["positive electrode"],
                auxiliary_domains={"secondary": "current collector"},
            ),
        )
        average_conc_broad = pybamm.x_average(conc_broad)
        self.assertIsInstance(average_conc_broad, pybamm.PrimaryBroadcast)
        self.assertEqual(average_conc_broad.domain, ["current collector"])
        conc_broad = pybamm.concatenation(
            pybamm.FullBroadcast(
                1,
                ["negative electrode"],
                auxiliary_domains={
                    "secondary": "current collector",
                    "tertiary": "test",
                },
            ),
            pybamm.FullBroadcast(
                2,
                ["separator"],
                auxiliary_domains={
                    "secondary": "current collector",
                    "tertiary": "test",
                },
            ),
            pybamm.FullBroadcast(
                3,
                ["positive electrode"],
                auxiliary_domains={
                    "secondary": "current collector",
                    "tertiary": "test",
                },
            ),
        )
        average_conc_broad = pybamm.x_average(conc_broad)
        self.assertIsInstance(average_conc_broad, pybamm.FullBroadcast)
        assert_domain_equal(
            average_conc_broad.domains,
            {"primary": ["current collector"], "secondary": ["test"]},
        )

        # x-average of broadcast
        for domain in [["negative electrode"], ["separator"], ["positive electrode"]]:
            a = pybamm.Variable("a", domain=domain)
            x = pybamm.SpatialVariable("x", domain)
            av_a = pybamm.x_average(a)
            self.assertIsInstance(av_a, pybamm.XAverage)
            self.assertEqual(av_a.integration_variable[0].domain, x.domain)
            self.assertEqual(av_a.domain, [])

        # whole electrode domain
        domain = ["negative electrode", "separator", "positive electrode"]
        a = pybamm.Variable("a", domain=domain)
        x = pybamm.SpatialVariable("x", domain)
        av_a = pybamm.x_average(a)
        self.assertIsInstance(av_a, pybamm.XAverage)
        self.assertEqual(av_a.integration_variable[0].domain, x.domain)
        self.assertEqual(av_a.domain, [])

        a = pybamm.Variable("a", domain="new domain")
        av_a = pybamm.x_average(a)
        self.assertEqual(av_a, a)

        # x-average of symbol that evaluates on edges raises error
        symbol_on_edges = pybamm.SpatialVariableEdge(
            "x_n", domain=["negative electrode"]
        )
        with self.assertRaisesRegex(
            ValueError, "Can't take the x-average of a symbol that evaluates on edges"
        ):
            pybamm.x_average(symbol_on_edges)

        # Particle domains
        a = pybamm.Symbol(
            "a",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        av_a = pybamm.x_average(a)
        self.assertEqual(a.domain, ["negative particle"])
        self.assertIsInstance(av_a, pybamm.XAverage)

        a = pybamm.Symbol(
            "a",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )
        av_a = pybamm.x_average(a)
        self.assertEqual(a.domain, ["positive particle"])
        self.assertIsInstance(av_a, pybamm.XAverage)

        # Addition or Subtraction
        a = pybamm.Variable("a", domain="domain")
        b = pybamm.Variable("b", domain="domain")
        self.assertEqual(
            pybamm.x_average(a + b), pybamm.x_average(a) + pybamm.x_average(b)
        )
        self.assertEqual(
            pybamm.x_average(a - b), pybamm.x_average(a) - pybamm.x_average(b)
        )

    def test_size_average(self):
        # no domain
        a = pybamm.Scalar(1)
        average_a = pybamm.size_average(a)
        self.assertEqual(average_a, a)

        b = pybamm.FullBroadcast(
            1,
            ["negative particle"],
            {"secondary": "negative electrode", "tertiary": "current collector"},
        )
        # no "particle size" domain
        average_b = pybamm.size_average(b)
        self.assertEqual(average_b, b)

        # primary or secondary broadcast to "particle size" domain
        average_a = pybamm.size_average(
            pybamm.PrimaryBroadcast(a, "negative particle size")
        )
        self.assertEqual(average_a.evaluate(), np.array([1]))

        a = pybamm.Symbol("a", domain="negative particle")
        average_a = pybamm.size_average(
            pybamm.SecondaryBroadcast(a, "negative particle size")
        )
        self.assertEqual(average_a, a)

        for domain in [["negative particle size"], ["positive particle size"]]:
            a = pybamm.Symbol("a", domain=domain)
            R = pybamm.SpatialVariable("R", domain)
            av_a = pybamm.size_average(a)
            self.assertIsInstance(av_a, pybamm.SizeAverage)
            self.assertEqual(av_a.integration_variable[0].domain, R.domain)
            # domain list should now be empty
            self.assertEqual(av_a.domain, [])

        # R-average of symbol that evaluates on edges raises error
        symbol_on_edges = pybamm.PrimaryBroadcastToEdges(1, "domain")
        with self.assertRaisesRegex(
            ValueError,
            """Can't take the size-average of a symbol that evaluates on edges""",
        ):
            pybamm.size_average(symbol_on_edges)

    def test_r_average(self):
        a = pybamm.Scalar(1)
        average_a = pybamm.r_average(a)
        self.assertEqual(average_a, a)

        average_broad_a = pybamm.r_average(
            pybamm.PrimaryBroadcast(a, ["negative particle"])
        )
        self.assertEqual(average_broad_a.evaluate(), np.array([1]))

        for domain in [["negative particle"], ["positive particle"]]:
            a = pybamm.Symbol("a", domain=domain)
            r = pybamm.SpatialVariable("r", domain)
            av_a = pybamm.r_average(a)
            self.assertIsInstance(av_a, pybamm.RAverage)
            self.assertEqual(av_a.integration_variable[0].domain, r.domain)
            # electrode domains go to current collector when averaged
            self.assertEqual(av_a.domain, [])

        # r-average of a symbol that is broadcast to x
        # takes the average of the child then broadcasts it
        a = pybamm.PrimaryBroadcast(1, "positive particle")
        broad_a = pybamm.SecondaryBroadcast(a, "positive electrode")
        average_broad_a = pybamm.r_average(broad_a)
        self.assertIsInstance(average_broad_a, pybamm.PrimaryBroadcast)
        self.assertEqual(average_broad_a.domain, ["positive electrode"])
        self.assertEqual(average_broad_a.children[0], pybamm.r_average(a))

        # r-average of symbol that evaluates on edges raises error
        symbol_on_edges = pybamm.PrimaryBroadcastToEdges(1, "domain")
        with self.assertRaisesRegex(
            ValueError, "Can't take the r-average of a symbol that evaluates on edges"
        ):
            pybamm.r_average(symbol_on_edges)

        # Addition or Subtraction
        a = pybamm.Variable("a", domain="domain")
        b = pybamm.Variable("b", domain="domain")
        self.assertEqual(
            pybamm.r_average(a + b), pybamm.r_average(a) + pybamm.r_average(b)
        )
        self.assertEqual(
            pybamm.r_average(a - b), pybamm.r_average(a) - pybamm.r_average(b)
        )

    def test_yz_average(self):
        a = pybamm.Scalar(1)
        z_average_a = pybamm.z_average(a)
        yz_average_a = pybamm.yz_average(a)
        self.assertEqual(z_average_a, a)
        self.assertEqual(yz_average_a, a)

        z_average_broad_a = pybamm.z_average(
            pybamm.PrimaryBroadcast(a, ["current collector"])
        )
        yz_average_broad_a = pybamm.yz_average(
            pybamm.PrimaryBroadcast(a, ["current collector"])
        )
        self.assertEqual(z_average_broad_a.evaluate(), np.array([1]))
        self.assertEqual(yz_average_broad_a.evaluate(), np.array([1]))

        a = pybamm.Variable("a", domain=["current collector"])
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])

        yz_av_a = pybamm.yz_average(a)
        self.assertIsInstance(yz_av_a, pybamm.YZAverage)
        self.assertEqual(yz_av_a.integration_variable[0].domain, y.domain)
        self.assertEqual(yz_av_a.integration_variable[1].domain, z.domain)
        self.assertEqual(yz_av_a.domain, [])

        z_av_a = pybamm.z_average(a)
        self.assertIsInstance(z_av_a, pybamm.ZAverage)
        self.assertEqual(z_av_a.integration_variable[0].domain, a.domain)
        self.assertEqual(z_av_a.domain, [])

        a = pybamm.Symbol("a", domain="bad domain")
        with self.assertRaises(pybamm.DomainError):
            pybamm.z_average(a)
        with self.assertRaises(pybamm.DomainError):
            pybamm.yz_average(a)

        # average of symbol that evaluates on edges raises error
        symbol_on_edges = pybamm.PrimaryBroadcastToEdges(1, "domain")
        with self.assertRaisesRegex(
            ValueError, "Can't take the z-average of a symbol that evaluates on edges"
        ):
            pybamm.z_average(symbol_on_edges)

        # Addition or Subtraction
        a = pybamm.Variable("a", domain="current collector")
        b = pybamm.Variable("b", domain="current collector")
        self.assertEqual(
            pybamm.yz_average(a + b), pybamm.yz_average(a) + pybamm.yz_average(b)
        )
        self.assertEqual(
            pybamm.yz_average(a - b), pybamm.yz_average(a) - pybamm.yz_average(b)
        )
        self.assertEqual(
            pybamm.z_average(a + b), pybamm.z_average(a) + pybamm.z_average(b)
        )
        self.assertEqual(
            pybamm.z_average(a - b), pybamm.z_average(a) - pybamm.z_average(b)
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
