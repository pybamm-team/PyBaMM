#
# Tests for the Concatenation class and subclasses
#
import pybamm
import tests.shared as shared
import numpy as np
import unittest


class TestConcatenations(unittest.TestCase):
    def test_base_concatenation(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c")
        conc = pybamm.Concatenation(a, b, c)
        self.assertEqual(conc.name, "concatenation")
        self.assertIsInstance(conc.children[0], pybamm.Symbol)
        self.assertEqual(conc.children[0].name, "a")
        self.assertEqual(conc.children[1].name, "b")
        self.assertEqual(conc.children[2].name, "c")
        with self.assertRaises(NotImplementedError):
            conc.evaluate(None, 3)

    def test_concatenation_domains(self):
        a = pybamm.Symbol("a", domain=["negative electrode"])
        b = pybamm.Symbol("b", domain=["separator", "positive electrode"])
        c = pybamm.Symbol("c", domain=["test"])
        conc = pybamm.Concatenation(a, b, c)
        self.assertEqual(
            conc.domain,
            ["negative electrode", "separator", "positive electrode", "test"],
        )

        # Can't concatenate nodes with overlapping domains
        d = pybamm.Symbol("d", domain=["separator"])
        with self.assertRaises(pybamm.DomainError):
            pybamm.Concatenation(a, b, d)

        # ensure concatenated domains are sorted correctly
        conc = pybamm.Concatenation(c, a, b)
        self.assertEqual(
            conc.domain,
            ["negative electrode", "separator", "positive electrode", "test"],
        )

    def test_numpy_model_concatenation_vectors(self):
        # with entries
        y = np.linspace(0, 1, 15)
        a = pybamm.Vector(y[:5])
        b = pybamm.Vector(y[5:9])
        c = pybamm.Vector(y[9:])
        conc = pybamm.NumpyModelConcatenation(a, b, c)
        np.testing.assert_array_equal(conc.evaluate(None, y), y)
        # with y_slice
        a = pybamm.StateVector(slice(0, 10))
        b = pybamm.StateVector(slice(10, 15))
        c = pybamm.StateVector(slice(15, 23))
        conc = pybamm.NumpyModelConcatenation(a, b, c)
        y = np.linspace(0, 1, 23)
        np.testing.assert_array_equal(conc.evaluate(None, y), y)

    def test_numpy_model_concatenation_vector_scalar(self):
        # with entries
        y = np.linspace(0, 1, 10)
        a = pybamm.Vector(y)
        b = pybamm.Scalar(16)
        c = pybamm.Scalar(3)
        conc = pybamm.NumpyModelConcatenation(a, b, c)
        np.testing.assert_array_equal(
            conc.evaluate(None, y), np.concatenate([y, np.array([16]), np.array([3])])
        )

        # with y_slice
        a = pybamm.StateVector(slice(0, 10))
        conc = pybamm.NumpyModelConcatenation(a, b, c)
        np.testing.assert_array_equal(
            conc.evaluate(None, y), np.concatenate([y, np.array([16]), np.array([3])])
        )

    def test_numpy_domain_concatenation(self):
        # create discretisation
        defaults = shared.TestDefaults1DMacro()
        disc = shared.DiscretisationForTesting(defaults.mesh)
        mesh = disc.mesh

        a_dom = ["negative electrode"]
        b_dom = ["positive electrode"]
        a = pybamm.Scalar(2, domain=a_dom)
        b = pybamm.Vector(np.ones_like(mesh[b_dom[0]].nodes), domain=b_dom)

        # concatenate them the "wrong" way round to check they get reordered correctly
        conc = pybamm.DomainConcatenation([b, a], mesh)
        np.testing.assert_array_equal(
            conc.evaluate(),
            np.concatenate(
                [np.full(mesh[a_dom[0]].npts, 2), np.full(mesh[b_dom[0]].npts, 1)]
            ),
        )

        # vector child of wrong size will throw
        b = pybamm.Vector(np.full(mesh[b_dom[0]].npts - 5, 1), domain=b_dom)
        with self.assertRaises(ValueError):
            conc = pybamm.DomainConcatenation([b, a], mesh)

        # check the reordering in case a child vector has to be split up
        a_dom = ["separator"]
        b_dom = ["negative electrode", "positive electrode"]
        a = pybamm.Scalar(2, domain=a_dom)
        b = pybamm.Vector(
            np.concatenate(
                [np.full(mesh[b_dom[0]].npts, 1), np.full(mesh[b_dom[1]].npts, 3)]
            ),
            domain=b_dom,
        )

        conc = pybamm.DomainConcatenation([a, b], mesh)
        np.testing.assert_array_equal(
            conc.evaluate(),
            np.concatenate(
                [
                    np.full(mesh[b_dom[0]].npts, 1),
                    np.full(mesh[a_dom[0]].npts, 2),
                    np.full(mesh[b_dom[1]].npts, 3),
                ]
            ),
        )

    def test_concatenation_orphans(self):
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        c = pybamm.Variable("c")
        conc = pybamm.Concatenation(a, b, c)
        a_new, b_new, c_new = conc.orphans

        # We should be able to manipulate the children without TreeErrors
        self.assertIsInstance(2 * a_new, pybamm.Multiplication)
        self.assertIsInstance(3 + b_new, pybamm.Addition)
        self.assertIsInstance(4 - c_new, pybamm.Subtraction)

        # ids should stay the same
        self.assertEqual(a.id, a_new.id)
        self.assertEqual(b.id, b_new.id)
        self.assertEqual(c.id, c_new.id)
        self.assertEqual(conc.id, pybamm.Concatenation(a_new, b_new, c_new).id)


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
