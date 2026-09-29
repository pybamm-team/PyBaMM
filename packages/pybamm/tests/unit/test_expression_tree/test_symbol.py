#
# Test for the Symbol class
#

import pickle  # nosec B403 - used in tests with trusted input
import re

import numpy as np
import pytest
import sympy
from scipy.sparse import coo_matrix, csr_matrix

import pybamm
from pybamm.expression_tree.binary_operators import _Heaviside
from pybamm.expression_tree.symbol import domain_size


class TestDomainSize:
    def test_empty_domain(self):
        assert domain_size([]) == 1
        assert domain_size(None) == 1

    def test_fixed_domains(self):
        assert domain_size(["negative electrode"]) == 11
        assert domain_size(["separator"]) == 13
        assert domain_size(["positive electrode"]) == 17

    def test_fixed_domains_are_additive(self):
        assert domain_size(["negative electrode", "separator"]) == 11 + 13
        assert (
            domain_size(["negative electrode", "separator", "positive electrode"])
            == 11 + 13 + 17
        )

    def test_custom_domains_are_additive(self):
        size_a = domain_size(["domain_a"])
        size_b = domain_size(["domain_b"])
        assert domain_size(["domain_a", "domain_b"]) == size_a + size_b

    def test_custom_domain_size_at_least_2(self):
        assert domain_size(["any_custom_domain"]) >= 2


class TestSymbol:
    def test_symbol_init(self):
        sym = pybamm.Symbol("a symbol")
        with pytest.raises(TypeError):
            pybamm.Symbol(1)
        with pytest.raises(AttributeError):
            sym.name = "renamed"
        assert sym.name == "a symbol"
        assert str(sym) == "a symbol"

    def test_children(self):
        symc1 = pybamm.Symbol("child1")
        symc2 = pybamm.Symbol("child2")
        symp = pybamm.Symbol("parent", children=[symc1, symc2])

        # test tuples of children for equality based on their name
        def check_are_equal(children1, children2):
            assert len(children1) == len(children2)
            for i in range(len(children1)):
                assert children1[i].name == children2[i].name

        check_are_equal(symp.children, (symc1, symc2))

    def test_symbol_domains(self):
        a = pybamm.Symbol("a", domain="test")
        assert a.domain == ["test"]
        # domains are fixed at construction
        with pytest.raises(AttributeError):
            a.domains = {"primary": ["test"]}
        assert a.domains["primary"] == ["test"]
        a = pybamm.Symbol("a", domain=["t", "e", "s"])
        assert a.domain == ["t", "e", "s"]
        with pytest.raises(TypeError):
            a = pybamm.Symbol("a", domain=1)
        with pytest.raises(
            pybamm.DomainError,
            match=r"Domain levels must be filled in order",
        ):
            b = pybamm.Symbol("b", auxiliary_domains={"secondary": ["test sec"]})
        b = pybamm.Symbol(
            "b", domain="test", auxiliary_domains={"secondary": ["test sec"]}
        )

        with pytest.raises(pybamm.DomainError, match=r"keys must be one of"):
            pybamm.Symbol("b", domains={"test": "test"})
        with pytest.raises(ValueError, match=r"Only one of 'domain' or 'domains'"):
            pybamm.Symbol("b", domain="test", domains={"primary": "test"})
        with pytest.raises(
            ValueError, match=r"Only one of 'auxiliary_domains' or 'domains'"
        ):
            pybamm.Symbol(
                "b",
                auxiliary_domains={"secondary": "other test"},
                domains={"test": "test"},
            )
        with pytest.raises(NotImplementedError, match=r"Cannot set domain directly"):
            b.domain = "test"

    def test_symbol_auxiliary_domains(self):
        a = pybamm.Symbol(
            "a",
            domain="test",
            auxiliary_domains={
                "secondary": "sec",
                "tertiary": "tert",
                "quaternary": "quat",
            },
        )
        assert a.domain == ["test"]
        assert a.secondary_domain == ["sec"]
        assert a.tertiary_domain == ["tert"]
        assert a.tertiary_domain == ["tert"]
        assert a.quaternary_domain == ["quat"]
        assert a.domains == {
            "primary": ["test"],
            "secondary": ["sec"],
            "tertiary": ["tert"],
            "quaternary": ["quat"],
        }

        a = pybamm.Symbol("a", domain=["t", "e", "s"])
        assert a.domain == ["t", "e", "s"]
        with pytest.raises(TypeError):
            a = pybamm.Symbol("a", domain=1)
        with pytest.raises(pybamm.DomainError, match=r"All domains must be different"):
            pybamm.Symbol("b", domains={"primary": "test", "secondary": "test"})
        with pytest.raises(pybamm.DomainError, match=r"All domains must be different"):
            pybamm.Symbol(
                "b",
                domain="test",
                auxiliary_domains={"secondary": ["test sec"], "tertiary": ["test sec"]},
            )

        with pytest.raises(NotImplementedError, match=r"auxiliary_domains"):
            a.auxiliary_domains

    def test_symbol_methods(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")

        # unary
        assert isinstance(-a, pybamm.Negate)
        assert isinstance(abs(a), pybamm.AbsoluteValue)
        # special cases
        assert -(-a) == a  # noqa: B002
        assert -(a - b) == b - a
        assert abs(abs(a)) == abs(a)

        # binary - two symbols
        assert isinstance(a + b, pybamm.Addition)
        assert isinstance(a - b, pybamm.Subtraction)
        assert isinstance(a * b, pybamm.Multiplication)
        assert isinstance(a @ b, pybamm.MatrixMultiplication)
        assert isinstance(a / b, pybamm.Division)
        assert isinstance(a**b, pybamm.Power)
        assert isinstance(a < b, _Heaviside)
        assert isinstance(a <= b, _Heaviside)
        assert isinstance(a > b, _Heaviside)
        assert isinstance(a >= b, _Heaviside)
        assert isinstance(a % b, pybamm.Modulo)

        # binary - symbol and number
        assert isinstance(a + 2, pybamm.Addition)
        assert isinstance(2 - a, pybamm.Subtraction)
        assert isinstance(a * 2, pybamm.Multiplication)
        assert isinstance(a @ 2, pybamm.MatrixMultiplication)
        assert isinstance(2 / a, pybamm.Division)
        assert isinstance(a**2, pybamm.Power)

        # binary - number and symbol
        assert isinstance(3 + b, pybamm.Addition)
        assert (3 + b).children[1] == b
        assert isinstance(3 - b, pybamm.Subtraction)
        assert (3 - b).children[1] == b
        assert isinstance(3 * b, pybamm.Multiplication)
        assert (3 * b).children[1] == b
        assert isinstance(3 @ b, pybamm.MatrixMultiplication)
        assert (3 @ b).children[1] == b
        assert isinstance(3 / b, pybamm.Division)
        assert (3 / b).children[1] == b
        assert isinstance(3**b, pybamm.Power)
        assert (3**b).children[1] == b

        # error raising
        with pytest.raises(
            NotImplementedError,
            match=r"BinaryOperator not implemented for symbols of type",
        ):
            a + "two"

    def test_symbol_create_copy(self):
        a = pybamm.Symbol("a")
        new_a = a.create_copy()
        assert new_a == a

        b = pybamm.Symbol("b")
        new_b = b.create_copy(new_children=[a])
        assert new_b == pybamm.Symbol("b", children=[a])

    def test_sigmoid(self):
        # Test that smooth heaviside is used when the setting is changed
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")

        pybamm.settings.heaviside_smoothing = 10

        assert str(a < b) == str(pybamm.sigmoid(a, b, 10))
        assert str(a <= b) == str(pybamm.sigmoid(a, b, 10))
        assert str(a > b) == str(pybamm.sigmoid(b, a, 10))
        assert str(a >= b) == str(pybamm.sigmoid(b, a, 10))

        # But exact heavisides should still be used if both variables are constant
        a = pybamm.Scalar(1)
        b = pybamm.Scalar(2)
        assert str(a < b) == str(pybamm.Scalar(1))
        assert str(a <= b) == str(pybamm.Scalar(1))
        assert str(a > b) == str(pybamm.Scalar(0))
        assert str(a >= b) == str(pybamm.Scalar(0))

        # Change setting back for other tests
        pybamm.settings.heaviside_smoothing = "exact"

    def test_smooth_absolute_value(self):
        # Test that smooth absolute value is used when the setting is changed
        a = pybamm.Symbol("a")
        pybamm.settings.abs_smoothing = 10
        assert str(abs(a)) == str(pybamm.smooth_absolute_value(a, 10))

        # But exact absolute value should still be used for constants
        a = pybamm.Scalar(-5)
        assert str(abs(a)) == str(pybamm.Scalar(5))

        # Change setting back for other tests
        pybamm.settings.abs_smoothing = "exact"

    def test_multiple_symbols(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c")
        exp = a * c * (a * b * c + a - c * a)
        expected_preorder = [
            "*",
            "*",
            "a",
            "c",
            "-",
            "+",
            "*",
            "*",
            "a",
            "b",
            "c",
            "a",
            "*",
            "c",
            "a",
        ]
        expected_postorder = [
            "a",
            "c",
            "*",
            "a",
            "b",
            "*",
            "c",
            "*",
            "a",
            "+",
            "c",
            "a",
            "*",
            "-",
            "*",
        ]
        for node, expect in zip(exp.pre_order(), expected_preorder, strict=False):
            assert node.name == expect
        for node, expect in zip(exp.post_order(), expected_postorder, strict=False):
            assert node.name == expect

    def test_symbol_diff(self):
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        assert isinstance(a.diff(a), pybamm.Scalar)
        assert a.diff(a).evaluate() == 1
        assert isinstance(a.diff(b), pybamm.Scalar)
        assert a.diff(b).evaluate() == 0

    def test_symbol_evaluation(self):
        a = pybamm.Symbol("a")
        with pytest.raises(NotImplementedError):
            a.evaluate()

    def test_evaluate_ignoring_errors(self):
        assert pybamm.t.evaluate_ignoring_errors(t=None) is None
        assert pybamm.t.evaluate_ignoring_errors(t=0) == 0
        assert pybamm.Parameter("a").evaluate_ignoring_errors() is None
        assert pybamm.StateVector(slice(0, 1)).evaluate_ignoring_errors() is None
        assert pybamm.StateVectorDot(slice(0, 1)).evaluate_ignoring_errors() is None

        np.testing.assert_array_equal(
            pybamm.InputParameter("a").evaluate_ignoring_errors(), np.nan
        )

    def test_symbol_is_constant(self):
        a = pybamm.Variable("a")
        assert not a.is_constant()

        a = pybamm.Parameter("a")
        assert not a.is_constant()

        a = pybamm.Scalar(1) * pybamm.Variable("a")
        assert not a.is_constant()

        a = pybamm.Scalar(1) * pybamm.StateVector(slice(10))
        assert not a.is_constant()

        a = pybamm.Scalar(1) * pybamm.Vector(np.zeros(10))
        assert a.is_constant()

    def test_symbol_evaluates_to_number(self):
        a = pybamm.Scalar(3)
        assert a.evaluates_to_number()

        a = pybamm.Parameter("a")
        assert a.evaluates_to_number()

        a = pybamm.Scalar(3) * pybamm.Time()
        assert a.evaluates_to_number()
        # highlight difference between this function and isinstance(a, Scalar)
        assert not isinstance(a, pybamm.Scalar)

        a = pybamm.Variable("a")
        assert not a.evaluates_to_number()

        a = pybamm.Scalar(3) - 2
        assert a.evaluates_to_number()

        a = pybamm.Vector(np.ones(5))
        assert not a.evaluates_to_number()

        a = pybamm.Matrix(np.ones((4, 6)))
        assert not a.evaluates_to_number()

        a = pybamm.StateVector(slice(0, 10))
        assert not a.evaluates_to_number()

        # Time variable returns false
        a = 3 * pybamm.t + 2
        assert a.evaluates_to_number()

    def test_symbol_evaluates_to_constant_number(self):
        a = pybamm.Scalar(3)
        assert a.evaluates_to_constant_number()

        a = pybamm.Parameter("a")
        assert not a.evaluates_to_constant_number()

        a = pybamm.Variable("a")
        assert not a.evaluates_to_constant_number()

        a = pybamm.Scalar(3) - 2
        assert a.evaluates_to_constant_number()

        a = pybamm.Vector(np.ones(5))
        assert not a.evaluates_to_constant_number()

        a = pybamm.Matrix(np.ones((4, 6)))
        assert not a.evaluates_to_constant_number()

        a = pybamm.StateVector(slice(0, 10))
        assert not a.evaluates_to_constant_number()

        # Time variable returns true
        a = 3 * pybamm.t + 2
        assert not a.evaluates_to_constant_number()

    def test_simplify_if_constant(self):
        m = pybamm.Matrix(np.zeros((10, 10)))
        m_simp = pybamm.simplify_if_constant(m)
        assert isinstance(m_simp, pybamm.Matrix)
        assert isinstance(m_simp.entries, csr_matrix)

    def test_symbol_repr(self):
        """
        test that __repr___ returns the string
        `__class__(id, name, children, domain, auxiliary_domains)`
        """
        a = pybamm.Symbol("a")
        b = pybamm.Symbol("b")
        c = pybamm.Symbol("c", domain=["test"])
        d = pybamm.Symbol(
            "d", domain=["test"], auxiliary_domains={"secondary": "other test"}
        )
        hex_regex = r"\-?0x[0-9,a-f]+"
        assert re.search(
            r"Symbol\(" + hex_regex + r", a, children=\[\], domains=\{\}\)",
            a.__repr__(),
        )
        assert re.search(
            r"Symbol\(" + hex_regex + r", b, children=\[\], domains=\{\}\)",
            b.__repr__(),
        )

        assert re.search(
            r"Symbol\("
            + hex_regex
            + r", c, children=\[\], domains=\{'primary': \['test'\]\}\)",
            c.__repr__(),
        )

        assert re.search(
            r"Symbol\("
            + hex_regex
            + r", d, children=\[\], domains=\{'primary': \['test'\], "
            + r"'secondary': \['other test'\]\}\)",
            d.__repr__(),
        )

        assert re.search(
            r"Addition\(" + hex_regex + r", \+, children=\['a', 'b'\], domains=\{\}\)",
            (a + b).__repr__(),
        )

        assert re.search(
            r"Multiplication\("
            + hex_regex
            + r", \*, children=\['a', 'd'\], domains=\{'primary': \['test'\], "
            + r"'secondary': \['other test'\]\}\)",
            (a * d).__repr__(),
        )

        assert re.search(
            r"Gradient\("
            + hex_regex
            + r", grad, children=\['c'\], domains=\{'primary': \['test'\]\}\)",
            pybamm.grad(c).__repr__(),
        )

    @pytest.fixture(scope="session")
    def test_symbol_visualise(self, tmp_path):
        temp_file = tmp_path / "test_visualize.png"
        c = pybamm.Variable("c", "negative electrode")
        d = pybamm.Variable("d", "negative electrode")
        sym = pybamm.div(c * pybamm.grad(c)) + (c / d + c - d) ** 5
        sym.visualise(str(temp_file))
        assert temp_file.exists()

        with pytest.raises(ValueError):
            sym.visualise(str(temp_file.with_suffix("")))

    def test_has_spatial_derivatives(self):
        var = pybamm.Variable("var", domain="test")
        grad_eqn = pybamm.grad(var)
        div_eqn = pybamm.div(pybamm.standard_spatial_vars.x_edge)
        grad_div_eqn = pybamm.div(grad_eqn)
        algebraic_eqn = 2 * var + 3
        assert grad_eqn.has_symbol_of_classes(pybamm.Gradient)
        assert not grad_eqn.has_symbol_of_classes(pybamm.Divergence)
        assert not div_eqn.has_symbol_of_classes(pybamm.Gradient)
        assert div_eqn.has_symbol_of_classes(pybamm.Divergence)
        assert grad_div_eqn.has_symbol_of_classes(pybamm.Gradient)
        assert grad_div_eqn.has_symbol_of_classes(pybamm.Divergence)
        assert not algebraic_eqn.has_symbol_of_classes(pybamm.Gradient)
        assert not algebraic_eqn.has_symbol_of_classes(pybamm.Divergence)

    def test_orphans(self):
        a = pybamm.Scalar(1)
        b = pybamm.Parameter("b")
        summ = a + b

        a_orp, b_orp = summ.orphans
        assert a == a_orp
        assert b == b_orp

    def test_shape(self):
        scal = pybamm.Scalar(1)
        assert scal.shape == ()
        assert scal.size == 1

        state = pybamm.StateVector(slice(10))
        assert state.shape == (10, 1)
        assert state.size == 10
        state = pybamm.StateVector(slice(10, 25))
        assert state.shape == (15, 1)

        # test with big object
        state = 2 * pybamm.StateVector(slice(100000))
        assert state.shape == (100000, 1)

    def test_shape_and_size_for_testing(self):
        scal = pybamm.Scalar(1)
        assert scal.shape_for_testing == scal.shape
        assert scal.size_for_testing == scal.size

        state = pybamm.StateVector(slice(10, 25), domain="test")
        state2 = pybamm.StateVector(slice(10, 25), domain="test 2")
        assert state.shape_for_testing == state.shape

        param = pybamm.Parameter("a")
        assert param.shape_for_testing == ()

        func = pybamm.FunctionParameter("func", {"state": state})
        assert func.shape_for_testing == state.shape_for_testing

        concat = pybamm.concatenation(state, state2)
        assert concat.shape_for_testing == (30, 1)
        assert concat.size_for_testing == 30

        var = pybamm.Variable("var", domain="negative electrode")
        broadcast = pybamm.PrimaryBroadcast(0, "negative electrode")
        assert var.shape_for_testing == broadcast.shape_for_testing
        assert (var + broadcast).shape_for_testing == broadcast.shape_for_testing

        var = pybamm.Variable("var", domain=["random domain", "other domain"])
        broadcast = pybamm.PrimaryBroadcast(0, ["random domain", "other domain"])
        assert var.shape_for_testing == broadcast.shape_for_testing
        assert (var + broadcast).shape_for_testing == broadcast.shape_for_testing

        sym = pybamm.Symbol("sym")
        with pytest.raises(NotImplementedError):
            sym.shape_for_testing

    def test_test_shape(self):
        # right shape, passes
        y1 = pybamm.StateVector(slice(0, 10))
        y1.test_shape()
        # bad shape, fails
        y2 = pybamm.StateVector(slice(0, 5))
        with pytest.raises(pybamm.ShapeError):
            (y1 + y2).test_shape()

    def test_to_equation(self):
        assert pybamm.Symbol("test").to_equation() == sympy.Symbol("test")

    def test_numpy_array_ufunc(self):
        x = pybamm.Symbol("x")
        assert np.exp(x) == pybamm.exp(x)

    def test_setstate_refreshes_id(self, monkeypatch):
        # #5444: string hashes differ per process, so unpickling must drop the id
        var = pybamm.Variable("test_var")
        expected_hash = hash(var)

        # Simulate a stale _id pickled from a different process. Using
        # monkeypatch with a string attribute name avoids touching the
        # private member directly.
        monkeypatch.setattr(var, "_id", 12345)
        pickled = pickle.dumps(var)

        loaded = pickle.loads(pickled)  # nosec B301
        assert hash(loaded) == expected_hash
        assert hash(loaded) == hash(pybamm.Variable("test_var"))

    def test_id_is_computed_once(self):
        a = pybamm.Variable("a", domain="negative electrode")
        b = pybamm.Variable("b", domain="negative electrode")
        expr = a * b + pybamm.Scalar(2)
        first = expr.id
        assert expr.id == first
        assert hash(expr) == first
        assert expr == a * b + pybamm.Scalar(2)

    def test_immutable_once_hashed(self):
        var = pybamm.Variable("var", domain="negative electrode")
        hash(var)
        # legacy identity setters are forbidden throughout the test suite
        with pytest.raises(AttributeError):
            var.name = "other"
        with pytest.raises(AttributeError):
            var.domains = {"primary": ["separator"]}
        # validation cannot bypass the mutation guard
        for attr in ("scale", "reference", "bounds"):
            with pytest.raises(AttributeError):
                setattr(var, attr, 1)
        # and arbitrary attributes cannot be attached
        with pytest.raises(AttributeError):
            var.arbitrary = 1
        assert not hasattr(var, "__dict__")

    def test_annotations_are_read_only(self):
        a = pybamm.Variable("a", domain="negative electrode")
        for attr in ("mesh", "secondary_mesh", "tertiary_mesh"):
            assert getattr(a, attr) is None
            with pytest.raises(AttributeError):
                setattr(a, attr, 1)
        vf = pybamm.VectorField(pybamm.Scalar(1), pybamm.Scalar(2))
        assert vf.disc_state_vector is None
        with pytest.raises(AttributeError):
            vf.disc_state_vector = 1

    def test_construction_across_threads(self):
        from concurrent.futures import ThreadPoolExecutor

        def build(i):
            return pybamm.Variable(f"v{i}", domain="negative electrode", scale=i + 1)

        with ThreadPoolExecutor(8) as executor:
            symbols = list(executor.map(build, range(200)))
        assert [symbol.scale.value for symbol in symbols] == list(range(1, 201))
        with pytest.raises(AttributeError, match=r"immutable once constructed"):
            symbols[0]._name = "changed"

    def test_edge_direction_is_part_of_identity(self):
        a = pybamm.Variable("a", domain="negative electrode")
        pinned = a._replace(_edges_direction="lr")
        assert pinned != a
        assert pinned.evaluates_on_edges("primary") == "lr"
        assert a.evaluates_on_edges("primary") is False

    def test_all_symbol_subclasses_define_slots(self):
        def subclasses(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from subclasses(sub)

        missing = [
            cls.__qualname__
            for cls in subclasses(pybamm.Symbol)
            if cls.__module__.startswith("pybamm.") and "__slots__" not in cls.__dict__
        ]
        assert missing == []

    def test_with_domains_returns_copy(self):
        a = pybamm.Variable("a", domain="negative electrode")
        expr = 2 * a
        id_before = expr.id
        cleared = expr.without_domains()
        assert cleared is not expr
        assert cleared.domains == {} or all(v == [] for v in cleared.domains.values())
        assert expr.id == id_before
        assert expr.domain == ["negative electrode"]
        assert cleared.id != expr.id
        # children are shared, not copied
        assert cleared.children[1] is expr.children[1]
        # same domains gives back the same object
        assert expr.with_domains(expr) is expr
        assert cleared.with_domains(a).domains == a.domains

    @pytest.mark.parametrize(
        "domains",
        [
            {"bogus": ["x"]},
            {"secondary": ["s"]},
            {"primary": ["x"], "secondary": ["x"]},
            pybamm.Domains({"bogus": ["x"]}),
            pybamm.Domains({"secondary": ["s"]}),
            pybamm.Domains({"primary": ["x"], "secondary": ["x"]}),
        ],
    )
    def test_with_domains_validates_domain_mapping(self, domains):
        with pytest.raises(pybamm.DomainError):
            pybamm.Variable("a").with_domains(domains)

    def test_domain_interning_is_not_public(self):
        assert not hasattr(pybamm, "intern_domains")

    def test_create_copy_with_new_scale(self):
        a = pybamm.Variable("a", domain="negative electrode", scale=2, reference=1)
        b = a.create_copy(scale=3)
        assert a.scale == 2 and b.scale == 3
        assert b.reference == a.reference
        assert a != b
        assert a.create_copy() == a

    def test_pickle_roundtrip_preserves_slots_and_id(self):
        a = pybamm.Variable("a", domain="negative electrode", scale=2)
        expr = pybamm.grad(a) * 3
        loaded = pickle.loads(pickle.dumps(expr))  # nosec B301
        assert loaded == expr
        (loaded_a,) = pybamm.SymbolUnpacker(pybamm.Variable).unpack_symbol(loaded)
        assert loaded_a.scale == 2
        assert not hasattr(loaded, "__dict__")

    def test_to_from_json(self, mocker):
        symc1 = pybamm.Symbol("child1", domain=["domain_1"])
        symc2 = pybamm.Symbol("child2", domain=["domain_2"])
        symp = pybamm.Symbol("parent", domain=["domain_3"], children=[symc1, symc2])

        json_dict = {
            "name": "parent",
            "domains": {
                "primary": ["domain_3"],
                "secondary": [],
                "tertiary": [],
                "quaternary": [],
            },
        }

        assert symp.to_json() == json_dict

        json_dict["children"] = [symc1, symc2]

        assert pybamm.Symbol._from_json(json_dict) == symp


class TestIsZero:
    def test_is_scalar_zero(self):
        a = pybamm.Scalar(0)
        b = pybamm.Scalar(2)
        assert pybamm.is_scalar_zero(a)
        assert not pybamm.is_scalar_zero(b)

    def test_is_matrix_zero(self):
        a = pybamm.Matrix(coo_matrix(np.zeros((10, 10))))
        b = pybamm.Matrix(coo_matrix(np.ones((10, 10))))
        c = pybamm.Matrix(coo_matrix(([1], ([0], [0])), shape=(5, 5)))
        assert pybamm.is_matrix_zero(a)
        assert not pybamm.is_matrix_zero(b)
        assert not pybamm.is_matrix_zero(c)

        a = pybamm.Matrix(np.zeros((10, 10)))
        b = pybamm.Matrix(np.ones((10, 10)))
        c = pybamm.Matrix([1, 0, 0])
        assert pybamm.is_matrix_zero(a)
        assert not pybamm.is_matrix_zero(b)
        assert not pybamm.is_matrix_zero(c)

    def test_bool(self):
        a = pybamm.Symbol("a")
        with pytest.raises(NotImplementedError, match=r"Boolean"):
            bool(a)
        # if statement calls Boolean
        with pytest.raises(NotImplementedError, match=r"Boolean"):
            if a > 1:
                print("a is greater than 1")
