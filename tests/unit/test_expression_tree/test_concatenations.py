#
# Tests for the Concatenation class and subclasses
#
import unittest
from tests import TestCase

import numpy as np
import sympy

import pybamm
from tests import get_discretisation_for_testing, get_mesh_for_testing


class TestConcatenations(TestCase):
    def test_base_concatenation(self):
        a = pybamm.Symbol("a", domain="test a")
        b = pybamm.Symbol("b", domain="test b")
        c = pybamm.Symbol("c", domain="test c")
        conc = pybamm.concatenation(a, b, c)
        self.assertEqual(conc.name, "concatenation")
        self.assertEqual(str(conc), "concatenation(a, b, c)")
        self.assertIsInstance(conc.children[0], pybamm.Symbol)
        self.assertEqual(conc.children[0].name, "a")
        self.assertEqual(conc.children[1].name, "b")
        self.assertEqual(conc.children[2].name, "c")
        d = pybamm.Vector([2], domain="test a")
        e = pybamm.Vector([1], domain="test b")
        f = pybamm.Vector([3], domain="test c")
        conc2 = pybamm.concatenation(d, e, f)
        with self.assertRaises(TypeError):
            conc2.evaluate()

        # trying to concatenate non-pybamm symbols
        with self.assertRaises(TypeError):
            pybamm.concatenation(1, 2)

        # concatenation of length 0
        with self.assertRaisesRegex(ValueError, "Cannot create empty concatenation"):
            pybamm.concatenation()

        # concatenation of lenght 1
        self.assertEqual(pybamm.concatenation(a), a)

        a = pybamm.Variable("a", domain="test a")
        b = pybamm.Variable("b", domain="test b")
        with self.assertRaisesRegex(TypeError, "ConcatenationVariable"):
            pybamm.Concatenation(a, b)

        # base concatenation jacobian
        a = pybamm.Symbol("a", domain="test a")
        b = pybamm.Symbol("b", domain="test b")
        conc3 = pybamm.Concatenation(a, b)
        with self.assertRaises(NotImplementedError):
            conc3._concatenation_jac(None)

    def test_concatenation_domains(self):
        a = pybamm.Symbol("a", domain=["negative electrode"])
        b = pybamm.Symbol("b", domain=["separator", "positive electrode"])
        c = pybamm.Symbol("c", domain=["test"])
        conc = pybamm.concatenation(a, b, c)
        self.assertEqual(
            conc.domain,
            ["negative electrode", "separator", "positive electrode", "test"],
        )

        # Can't concatenate nodes with overlapping domains
        d = pybamm.Symbol("d", domain=["separator"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.concatenation(a, b, d)

    def test_concatenation_auxiliary_domains(self):
        a = pybamm.Symbol(
            "a",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        b = pybamm.Symbol(
            "b",
            domain=["separator", "positive electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        conc = pybamm.concatenation(a, b)
        self.assertDomainEqual(
            conc.domains,
            {
                "primary": ["negative electrode", "separator", "positive electrode"],
                "secondary": ["current collector"],
            },
        )

        # Can't concatenate nodes with overlapping domains
        c = pybamm.Symbol(
            "c", domain=["test"], auxiliary_domains={"secondary": "something else"}
        )
        with self.assertRaisesRegex(
            pybamm.DomainError, "children must have same or empty auxiliary domains"
        ):
            pybamm.concatenation(a, b, c)

    def test_concatenations_scale(self):
        a = pybamm.Variable("a", domain="test a")
        b = pybamm.Variable("b", domain="test b")

        conc = pybamm.concatenation(a, b)
        self.assertEqual(conc.scale, 1)
        self.assertEqual(conc.reference, 0)

        a._scale = 2
        with self.assertRaisesRegex(
            ValueError, "Cannot concatenate symbols with different scales"
        ):
            pybamm.concatenation(a, b)

        b._scale = 2
        conc = pybamm.concatenation(a, b)
        self.assertEqual(conc.scale, 2)

        a._reference = 3
        with self.assertRaisesRegex(
            ValueError, "Cannot concatenate symbols with different references"
        ):
            pybamm.concatenation(a, b)

        b._reference = 3
        conc = pybamm.concatenation(a, b)
        self.assertEqual(conc.reference, 3)

        a.bounds = (0, 1)
        with self.assertRaisesRegex(
            ValueError, "Cannot concatenate symbols with different bounds"
        ):
            pybamm.concatenation(a, b)

        b.bounds = (0, 1)
        conc = pybamm.concatenation(a, b)
        self.assertEqual(conc.bounds, (0, 1))

    def test_concatenation_simplify(self):
        # Primary broadcast
        var = pybamm.Variable("var", "current collector")
        a = pybamm.PrimaryBroadcast(var, "negative electrode")
        b = pybamm.PrimaryBroadcast(var, "separator")
        c = pybamm.PrimaryBroadcast(var, "positive electrode")

        concat = pybamm.concatenation(a, b, c)
        self.assertIsInstance(concat, pybamm.PrimaryBroadcast)
        self.assertEqual(concat.orphans[0], var)
        self.assertEqual(
            concat.domain, ["negative electrode", "separator", "positive electrode"]
        )

        # Full broadcast
        a = pybamm.FullBroadcast(0, "negative electrode", "current collector")
        b = pybamm.FullBroadcast(0, "separator", "current collector")
        c = pybamm.FullBroadcast(0, "positive electrode", "current collector")

        concat = pybamm.concatenation(a, b, c)
        self.assertIsInstance(concat, pybamm.FullBroadcast)
        self.assertEqual(concat.orphans[0], pybamm.Scalar(0))
        self.assertDomainEqual(
            concat.domains,
            {
                "primary": ["negative electrode", "separator", "positive electrode"],
                "secondary": ["current collector"],
            },
        )

    def test_numpy_concatenation_vectors(self):
        # with entries
        y = np.linspace(0, 1, 15)[:, np.newaxis]
        a = pybamm.Vector(y[:5])
        b = pybamm.Vector(y[5:9])
        c = pybamm.Vector(y[9:])
        conc = pybamm.NumpyConcatenation(a, b, c)
        np.testing.assert_array_equal(conc.evaluate(None, y), y)
        # with y_slice
        a = pybamm.StateVector(slice(0, 10))
        b = pybamm.StateVector(slice(10, 15))
        c = pybamm.StateVector(slice(15, 23))
        conc = pybamm.NumpyConcatenation(a, b, c)
        y = np.linspace(0, 1, 23)[:, np.newaxis]
        np.testing.assert_array_equal(conc.evaluate(None, y), y)
        # empty concatenation
        conc = pybamm.NumpyConcatenation()
        self.assertEqual(conc._concatenation_jac(None), 0)

    def test_numpy_concatenation_vector_scalar(self):
        # with entries
        y = np.linspace(0, 1, 10)[:, np.newaxis]
        a = pybamm.Vector(y)
        b = pybamm.Scalar(16)
        c = pybamm.Scalar(3)
        conc = pybamm.NumpyConcatenation(a, b, c)
        np.testing.assert_array_equal(
            conc.evaluate(y=y), np.concatenate([y, np.array([[16]]), np.array([[3]])])
        )

        # with y_slice
        a = pybamm.StateVector(slice(0, 10))
        conc = pybamm.NumpyConcatenation(a, b, c)
        np.testing.assert_array_equal(
            conc.evaluate(y=y), np.concatenate([y, np.array([[16]]), np.array([[3]])])
        )

        # with time
        b = pybamm.t
        conc = pybamm.NumpyConcatenation(a, b, c)
        np.testing.assert_array_equal(
            conc.evaluate(16, y), np.concatenate([y, np.array([[16]]), np.array([[3]])])
        )

    def test_domain_concatenation_domains(self):
        mesh = get_mesh_for_testing()
        # ensure concatenated domains are sorted correctly
        a = pybamm.Symbol("a", domain=["negative electrode"])
        b = pybamm.Symbol("b", domain=["separator", "positive electrode"])
        conc = pybamm.DomainConcatenation([a, b], mesh)
        self.assertEqual(
            conc.domain,
            [
                "negative electrode",
                "separator",
                "positive electrode",
            ],
        )

        conc.secondary_dimensions_npts = 2
        with self.assertRaisesRegex(ValueError, "Concatenation and children must have"):
            conc.create_slices(None)

    def test_concatenation_orphans(self):
        a = pybamm.Variable("a", domain=["negative electrode"])
        b = pybamm.Variable("b", domain=["separator"])
        c = pybamm.Variable("c", domain=["positive electrode"])
        conc = pybamm.concatenation(a, b, c)
        a_new, b_new, c_new = conc.orphans

        # We should be able to manipulate the children without TreeErrors
        self.assertIsInstance(2 * a_new, pybamm.Multiplication)
        self.assertIsInstance(3 + b_new, pybamm.Addition)
        self.assertIsInstance(4 - c_new, pybamm.Subtraction)

        # ids should stay the same
        self.assertEqual(a, a_new)
        self.assertEqual(b, b_new)
        self.assertEqual(c, c_new)
        self.assertEqual(conc, pybamm.concatenation(a_new, b_new, c_new))

    def test_broadcast_and_concatenate(self):
        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        # Piecewise constant scalars
        a = pybamm.PrimaryBroadcast(1, ["negative electrode"])
        b = pybamm.PrimaryBroadcast(2, ["separator"])
        c = pybamm.PrimaryBroadcast(3, ["positive electrode"])
        conc = pybamm.concatenation(a, b, c)

        self.assertEqual(
            conc.domain, ["negative electrode", "separator", "positive electrode"]
        )
        self.assertEqual(conc.children[0].domain, ["negative electrode"])
        self.assertEqual(conc.children[1].domain, ["separator"])
        self.assertEqual(conc.children[2].domain, ["positive electrode"])
        processed_conc = disc.process_symbol(conc)
        np.testing.assert_array_equal(
            processed_conc.evaluate(),
            np.concatenate(
                [
                    np.ones(mesh["negative electrode"].npts),
                    2 * np.ones(mesh["separator"].npts),
                    3 * np.ones(mesh["positive electrode"].npts),
                ]
            )[:, np.newaxis],
        )

        # Piecewise constant functions of time
        a_t = pybamm.PrimaryBroadcast(pybamm.t, ["negative electrode"])
        b_t = pybamm.PrimaryBroadcast(2 * pybamm.t, ["separator"])
        c_t = pybamm.PrimaryBroadcast(3 * pybamm.t, ["positive electrode"])
        conc = pybamm.concatenation(a_t, b_t, c_t)

        self.assertEqual(
            conc.domain, ["negative electrode", "separator", "positive electrode"]
        )
        self.assertEqual(conc.children[0].domain, ["negative electrode"])
        self.assertEqual(conc.children[1].domain, ["separator"])
        self.assertEqual(conc.children[2].domain, ["positive electrode"])

        processed_conc = disc.process_symbol(conc)
        np.testing.assert_array_equal(
            processed_conc.evaluate(t=2),
            np.concatenate(
                [
                    2 * np.ones(mesh["negative electrode"].npts),
                    4 * np.ones(mesh["separator"].npts),
                    6 * np.ones(mesh["positive electrode"].npts),
                ]
            )[:, np.newaxis],
        )

        # Piecewise constant state vectors
        a_sv = pybamm.PrimaryBroadcast(
            pybamm.StateVector(slice(0, 1)), ["negative electrode"]
        )
        b_sv = pybamm.PrimaryBroadcast(pybamm.StateVector(slice(1, 2)), ["separator"])
        c_sv = pybamm.PrimaryBroadcast(
            pybamm.StateVector(slice(2, 3)), ["positive electrode"]
        )
        conc = pybamm.concatenation(a_sv, b_sv, c_sv)

        self.assertEqual(
            conc.domain, ["negative electrode", "separator", "positive electrode"]
        )
        self.assertEqual(conc.children[0].domain, ["negative electrode"])
        self.assertEqual(conc.children[1].domain, ["separator"])
        self.assertEqual(conc.children[2].domain, ["positive electrode"])

        processed_conc = disc.process_symbol(conc)
        y = np.array([1, 2, 3])
        np.testing.assert_array_equal(
            processed_conc.evaluate(y=y),
            np.concatenate(
                [
                    np.ones(mesh["negative electrode"].npts),
                    2 * np.ones(mesh["separator"].npts),
                    3 * np.ones(mesh["positive electrode"].npts),
                ]
            )[:, np.newaxis],
        )

        # Mixed
        conc = pybamm.concatenation(a, b_t, c_sv)

        self.assertEqual(
            conc.domain, ["negative electrode", "separator", "positive electrode"]
        )
        self.assertEqual(conc.children[0].domain, ["negative electrode"])
        self.assertEqual(conc.children[1].domain, ["separator"])
        self.assertEqual(conc.children[2].domain, ["positive electrode"])

        processed_conc = disc.process_symbol(conc)
        np.testing.assert_array_equal(
            processed_conc.evaluate(t=2, y=y),
            np.concatenate(
                [
                    np.ones(mesh["negative electrode"].npts),
                    4 * np.ones(mesh["separator"].npts),
                    3 * np.ones(mesh["positive electrode"].npts),
                ]
            )[:, np.newaxis],
        )

    def test_domain_error(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        with self.assertRaisesRegex(
            pybamm.DomainError, "Cannot concatenate child 'a' with empty domain"
        ):
            pybamm.DomainConcatenation([a, b], None)

    def test_numpy_concatenation(self):
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        c = pybamm.Variable("c")
        self.assertEqual(
            pybamm.numpy_concatenation(pybamm.numpy_concatenation(a, b), c),
            pybamm.NumpyConcatenation(a, b, c),
        )

    def test_to_equation(self):
        a = pybamm.Symbol("a", domain="test a")
        b = pybamm.Symbol("b", domain="test b")
        func_symbol = sympy.Symbol(r"\begin{cases}a\\b\end{cases}")

        # Test print_name
        func = pybamm.Concatenation(a, b)
        func.print_name = "test"
        self.assertEqual(func.to_equation(), sympy.Symbol("test"))

        # Test concat_sym
        self.assertEqual(pybamm.Concatenation(a, b).to_equation(), func_symbol)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
