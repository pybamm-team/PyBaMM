# Tests for the Concatenation class and subclasses
#
import pybamm

import numpy as np
import unittest


class MeshForTesting(pybamm.BaseMesh):
    def __init__(self):
        super().__init__(None)
        self["whole cell"] = self.submeshclass(np.linspace(0, 1, 100))
        self["negative electrode"] = self.submeshclass(self["whole cell"].nodes[:30])
        self["separator"] = self.submeshclass(self["whole cell"].nodes[30:40])
        self["positive electrode"] = self.submeshclass(self["whole cell"].nodes[40:])


class TestConcatenations(unittest.TestCase):
    def test_base_concatenation(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c")
        conc = pybamm.Concatenation(a, b, c)
        self.assertEqual(conc.name, "concatenation")
        self.assertTrue(isinstance(conc.children[0], pybamm.Symbol))
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

        # Whole cell concatenations should simplify
        conc = pybamm.Concatenation(a, b)
        self.assertEqual(conc.domain, ["whole cell"])

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

    def test_numpy_concatenation_vectors(self):
        # with entries
        y = np.linspace(0, 1, 15)
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
        y = np.linspace(0, 1, 23)
        np.testing.assert_array_equal(conc.evaluate(None, y), y)

    def test_numpy_concatenation_vector_scalar(self):
        # with entries
        y = np.linspace(0, 1, 10)
        a = pybamm.Vector(y)
        b = pybamm.Scalar(16)
        c = pybamm.Scalar(3)
        conc = pybamm.NumpyConcatenation(a, b, c)
        np.testing.assert_array_equal(
            conc.evaluate(None, y), np.concatenate([y, np.array([16]), np.array([3])])
        )

        # with y_slice
        a = pybamm.StateVector(slice(0, 10))
        conc = pybamm.NumpyConcatenation(a, b, c)
        np.testing.assert_array_equal(
            conc.evaluate(None, y), np.concatenate([y, np.array([16]), np.array([3])])
        )

    def test_numpy_domain_concatenation(self):
        mesh = MeshForTesting()
        a_dom = ["negative electrode"]
        b_dom = ["positive electrode"]
        a = pybamm.Scalar(2, domain=a_dom)
        b = pybamm.Vector(np.ones_like(mesh[b_dom[0]].nodes), domain=b_dom)

        # concatenate them the "wrong" way round to check they get reordered correctly
        conc = pybamm.NumpyDomainConcatenation([b, a], mesh)
        np.testing.assert_array_equal(
            conc.evaluate(),
            np.concatenate([
                np.full(mesh[a_dom[0]].npts, 2),
                np.full(mesh[b_dom[0]].npts, 1)
            ])
        )

        # vector child of wrong size will throw
        b = pybamm.Vector(np.full(mesh[b_dom[0]].npts - 5, 1), domain=b_dom)
        with self.assertRaises(ValueError):
            conc = pybamm.NumpyDomainConcatenation([b, a], mesh)

        # check the reordering in case a child vector has to be split up
        a_dom = ["separator"]
        b_dom = ["negative electrode", "positive electrode"]
        a = pybamm.Scalar(2, domain=a_dom)
        b = pybamm.Vector(
            np.concatenate([np.full(mesh[b_dom[0]].npts, 1),
                            np.full(mesh[b_dom[1]].npts, 3)]),
            domain=b_dom
        )

        conc = pybamm.NumpyDomainConcatenation([a, b], mesh)
        np.testing.assert_array_equal(
            conc.evaluate(),
            np.concatenate([
                np.full(mesh[b_dom[0]].npts, 1),
                np.full(mesh[a_dom[0]].npts, 2),
                np.full(mesh[b_dom[1]].npts, 3),
            ])
        )


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    unittest.main()
