#
# Tests for the Concatenation class and subclasses
#
import pytest
import unittest.mock as mock
from tests import assert_domain_equal


import numpy as np

import pybamm
import sympy
from tests import get_discretisation_for_testing, get_mesh_for_testing


class TestConcatenations:
    def test_base_concatenation(self):
        a = pybamm.Symbol("a", domain="test a")
        b = pybamm.Symbol("b", domain="test b")
        c = pybamm.Symbol("c", domain="test c")
        conc = pybamm.concatenation(a, b, c)
        assert conc.name == "concatenation"
        assert str(conc) == "concatenation(a, b, c)"
        assert isinstance(conc.children[0], pybamm.Symbol)
        assert conc.children[0].name == "a"
        assert conc.children[1].name == "b"
        assert conc.children[2].name == "c"
        d = pybamm.Vector([2], domain="test a")
        e = pybamm.Vector([1], domain="test b")
        f = pybamm.Vector([3], domain="test c")
        conc2 = pybamm.concatenation(d, e, f)
        with pytest.raises(TypeError):
            conc2.evaluate()

        # trying to concatenate non-pybamm symbols
        with pytest.raises(TypeError):
            pybamm.concatenation(1, 2)

        # concatenation of length 0
        with pytest.raises(ValueError, match="Cannot create empty concatenation"):
            pybamm.concatenation()

        # concatenation of lenght 1
        assert pybamm.concatenation(a) == a

        a = pybamm.Variable("a", domain="test a")
        b = pybamm.Variable("b", domain="test b")
        with pytest.raises(TypeError, match="ConcatenationVariable"):
            pybamm.Concatenation(a, b)

        # base concatenation jacobian
        a = pybamm.Symbol("a", domain="test a")
        b = pybamm.Symbol("b", domain="test b")
        conc3 = pybamm.Concatenation(a, b)
        with pytest.raises(NotImplementedError):
            conc3._concatenation_jac(None)

    def test_concatenation_domains(self):
        a = pybamm.Symbol("a", domain=["negative electrode"])
        b = pybamm.Symbol("b", domain=["separator", "positive electrode"])
        c = pybamm.Symbol("c", domain=["test"])
        conc = pybamm.concatenation(a, b, c)
        assert conc.domain == [
            "negative electrode",
            "separator",
            "positive electrode",
            "test",
        ]

        # Can't concatenate nodes with overlapping domains
        d = pybamm.Symbol("d", domain=["separator"])
        with pytest.raises(pybamm.DomainError):
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
        assert_domain_equal(
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
        with pytest.raises(
            pybamm.DomainError,
            match="children must have same or empty auxiliary domains",
        ):
            pybamm.concatenation(a, b, c)

    def test_concatenations_scale(self):
        a = pybamm.Variable("a", domain="test a")
        b = pybamm.Variable("b", domain="test b")

        conc = pybamm.concatenation(a, b)
        assert conc.scale == 1
        assert conc.reference == 0

        a._scale = 2
        with pytest.raises(
            ValueError, match="Cannot concatenate symbols with different scales"
        ):
            pybamm.concatenation(a, b)

        b._scale = 2
        conc = pybamm.concatenation(a, b)
        assert conc.scale == 2

        a._reference = 3
        with pytest.raises(
            ValueError, match="Cannot concatenate symbols with different references"
        ):
            pybamm.concatenation(a, b)

        b._reference = 3
        conc = pybamm.concatenation(a, b)
        assert conc.reference == 3

        a.bounds = (0, 1)
        with pytest.raises(
            ValueError, match="Cannot concatenate symbols with different bounds"
        ):
            pybamm.concatenation(a, b)

        b.bounds = (0, 1)
        conc = pybamm.concatenation(a, b)
        assert conc.bounds == (0, 1)

    def test_concatenation_simplify(self):
        # Primary broadcast
        var = pybamm.Variable("var", "current collector")
        a = pybamm.PrimaryBroadcast(var, "negative electrode")
        b = pybamm.PrimaryBroadcast(var, "separator")
        c = pybamm.PrimaryBroadcast(var, "positive electrode")

        concat = pybamm.concatenation(a, b, c)
        assert isinstance(concat, pybamm.PrimaryBroadcast)
        assert concat.orphans[0] == var
        assert concat.domain == [
            "negative electrode",
            "separator",
            "positive electrode",
        ]

        # Full broadcast
        a = pybamm.FullBroadcast(0, "negative electrode", "current collector")
        b = pybamm.FullBroadcast(0, "separator", "current collector")
        c = pybamm.FullBroadcast(0, "positive electrode", "current collector")

        concat = pybamm.concatenation(a, b, c)
        assert isinstance(concat, pybamm.FullBroadcast)
        assert concat.orphans[0] == pybamm.Scalar(0)
        assert_domain_equal(
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
        assert conc._concatenation_jac(None) == 0

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
        assert conc.domain == [
            "negative electrode",
            "separator",
            "positive electrode",
        ]

        conc.secondary_dimensions_npts = 2
        with pytest.raises(ValueError, match="Concatenation and children must have"):
            conc.create_slices(None)

    def test_concatenation_orphans(self):
        a = pybamm.Variable("a", domain=["negative electrode"])
        b = pybamm.Variable("b", domain=["separator"])
        c = pybamm.Variable("c", domain=["positive electrode"])
        conc = pybamm.concatenation(a, b, c)
        a_new, b_new, c_new = conc.orphans

        # We should be able to manipulate the children without TreeErrors
        assert isinstance(2 * a_new, pybamm.Multiplication)
        assert isinstance(3 + b_new, pybamm.Addition)
        assert isinstance(4 - c_new, pybamm.Subtraction)

        # ids should stay the same
        assert a == a_new
        assert b == b_new
        assert c == c_new
        assert conc == pybamm.concatenation(a_new, b_new, c_new)

    def test_broadcast_and_concatenate(self):
        # create discretisation
        disc = get_discretisation_for_testing()
        mesh = disc.mesh

        # Piecewise constant scalars
        a = pybamm.PrimaryBroadcast(1, ["negative electrode"])
        b = pybamm.PrimaryBroadcast(2, ["separator"])
        c = pybamm.PrimaryBroadcast(3, ["positive electrode"])
        conc = pybamm.concatenation(a, b, c)

        assert conc.domain == ["negative electrode", "separator", "positive electrode"]
        assert conc.children[0].domain == ["negative electrode"]
        assert conc.children[1].domain == ["separator"]
        assert conc.children[2].domain == ["positive electrode"]
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

        assert conc.domain == ["negative electrode", "separator", "positive electrode"]
        assert conc.children[0].domain == ["negative electrode"]
        assert conc.children[1].domain == ["separator"]
        assert conc.children[2].domain == ["positive electrode"]

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

        assert conc.domain == ["negative electrode", "separator", "positive electrode"]
        assert conc.children[0].domain == ["negative electrode"]
        assert conc.children[1].domain == ["separator"]
        assert conc.children[2].domain == ["positive electrode"]

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

        assert conc.domain == ["negative electrode", "separator", "positive electrode"]
        assert conc.children[0].domain == ["negative electrode"]
        assert conc.children[1].domain == ["separator"]
        assert conc.children[2].domain == ["positive electrode"]

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
        with pytest.raises(
            pybamm.DomainError, match="Cannot concatenate child 'a' with empty domain"
        ):
            pybamm.DomainConcatenation([a, b], None)

    def test_numpy_concatenation(self):
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        c = pybamm.Variable("c")
        assert pybamm.numpy_concatenation(
            pybamm.numpy_concatenation(a, b), c
        ) == pybamm.NumpyConcatenation(a, b, c)

    def test_to_equation(self):
        a = pybamm.Symbol("a", domain="test a")
        b = pybamm.Symbol("b", domain="test b")
        func_symbol = sympy.Symbol(r"\begin{cases}a\\b\end{cases}")

        # Test print_name
        func = pybamm.Concatenation(a, b)
        func.print_name = "test"
        assert func.to_equation() == sympy.Symbol("test")

        # Test concat_sym
        assert pybamm.Concatenation(a, b).to_equation() == func_symbol

    def test_to_from_json(self):
        # test DomainConcatenation
        mesh = get_mesh_for_testing()
        a = pybamm.Symbol("a", domain=["negative electrode"])
        b = pybamm.Symbol("b", domain=["separator", "positive electrode"])
        conc = pybamm.DomainConcatenation([a, b], mesh)

        json_dict = {
            "name": "domain_concatenation",
            "id": mock.ANY,
            "domains": {
                "primary": ["negative electrode", "separator", "positive electrode"],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
            "slices": {
                "negative electrode": [{"start": 0, "stop": 40, "step": None}],
                "separator": [{"start": 40, "stop": 65, "step": None}],
                "positive electrode": [{"start": 65, "stop": 100, "step": None}],
            },
            "size": 100,
            "children_slices": [
                {"negative electrode": [{"start": 0, "stop": 40, "step": None}]},
                {
                    "separator": [{"start": 0, "stop": 25, "step": None}],
                    "positive electrode": [{"start": 25, "stop": 60, "step": None}],
                },
            ],
            "secondary_dimensions_npts": 1,
        }

        assert conc.to_json() == json_dict

        # manually add children
        json_dict["children"] = [a, b]

        # check symbol re-creation
        assert pybamm.pybamm.DomainConcatenation._from_json(json_dict) == conc

        # -----------------------------
        # test NumpyConcatenation -----
        # -----------------------------

        y = np.linspace(0, 1, 15)[:, np.newaxis]
        a_np = pybamm.Vector(y[:5])
        b_np = pybamm.Vector(y[5:9])
        c_np = pybamm.Vector(y[9:])
        conc_np = pybamm.NumpyConcatenation(a_np, b_np, c_np)

        np_json = {
            "name": "numpy_concatenation",
            "id": mock.ANY,
            "domains": {
                "primary": [],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
        }

        # test to_json
        assert conc_np.to_json() == np_json

        # add children
        np_json["children"] = [a_np, b_np, c_np]

        # test _from_json
        assert pybamm.NumpyConcatenation._from_json(np_json) == conc_np
