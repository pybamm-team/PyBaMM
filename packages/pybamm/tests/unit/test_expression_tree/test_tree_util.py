#
# Tests for the flatten / unflatten protocol
#
import numpy as np
import pytest
from hypothesis import given, settings

import pybamm
from tests.strategies.symbols import symbols


def _roundtrip(symbol):
    leaves, treedef = pybamm.tree_flatten(symbol)
    rebuilt = pybamm.tree_unflatten(treedef, leaves)
    assert rebuilt is not symbol
    assert type(rebuilt) is type(symbol)
    assert rebuilt == symbol
    assert rebuilt.id == symbol.id
    assert rebuilt.children == symbol.children
    assert rebuilt.domains == symbol.domains
    assert rebuilt.name == symbol.name
    assert treedef.num_leaves == len(leaves)
    assert tuple(symbol.leaves) == leaves
    return rebuilt


class TestTreeUtil:
    def test_string_slots_are_one_state_field(self):
        class CustomSymbol(pybamm.Symbol):
            __slots__ = "_value"  # noqa: PLC0205 -- the string form is under test

            def __init__(self, value):
                super().__init__("custom")
                self._value = value

        first = CustomSymbol(1)
        second = CustomSymbol(2)
        assert first != second
        rebuilt = _roundtrip(first)
        assert rebuilt._value == 1

    @settings(max_examples=200, deadline=None)
    @given(symbols(max_leaves=6))
    def test_roundtrip_property(self, symbol):
        _roundtrip(symbol)

    def test_leaf_fields_are_leaves(self):
        a = pybamm.Variable(
            "a", domain="negative electrode", scale=2, reference=1, bounds=(0, 3)
        )
        leaves, treedef = pybamm.tree_flatten(a)
        assert leaves == (a.scale, a.reference, *a.bounds)
        assert treedef.n_children == 0
        rebuilt = _roundtrip(a)
        assert rebuilt.scale == 2 and rebuilt.bounds == (0, 3)

        x = pybamm.SpatialVariable("x", domain="negative electrode")
        integral = pybamm.Integral(a, x)
        leaves, _ = pybamm.tree_flatten(integral)
        assert leaves == (a, x)
        _roundtrip(integral)

        evaluate_at = pybamm.EvaluateAt(a, pybamm.Scalar(0.5))
        assert tuple(evaluate_at.leaves) == (a, evaluate_at.position)
        _roundtrip(evaluate_at)

        function_parameter = pybamm.FunctionParameter("f", {"a": a}, diff_variable=a)
        assert tuple(function_parameter.leaves) == (a, a)
        _roundtrip(function_parameter)
        no_diff = pybamm.FunctionParameter("f", {"a": a})
        assert tuple(no_diff.leaves) == (a,)
        _roundtrip(no_diff)

    def test_unflatten_with_new_leaves(self):
        a = pybamm.Variable("a", domain="negative electrode")
        b = pybamm.Variable("b", domain="negative electrode")
        expr = 2 * a + pybamm.grad(a)
        leaves, treedef = pybamm.tree_flatten(expr)
        # replace the second child only
        new_leaves = [leaves[0], pybamm.grad(b)]
        rebuilt = pybamm.tree_unflatten(treedef, new_leaves)
        assert rebuilt != expr
        assert rebuilt.right == pybamm.grad(b)
        # untouched leaves are shared, not copied
        assert rebuilt.children[0] is expr.children[0]

    def test_wrong_leaf_count(self):
        a = pybamm.Variable("a")
        leaves, treedef = pybamm.tree_flatten(2 * a)
        with pytest.raises(ValueError, match=r"expects 2 leaves, got 1"):
            pybamm.tree_unflatten(treedef, leaves[:1])
        assert "Multiplication" in repr(treedef)

    def test_aliases_follow_children(self):
        a, b = pybamm.Scalar(1), pybamm.Scalar(2)
        expr = pybamm.Multiplication(a, b)
        assert expr.left is expr.children[0] and expr.right is expr.children[1]
        neg = pybamm.Negate(a)
        assert neg.child is neg.children[0]
        vf = pybamm.VectorField(a, b)
        assert vf.components == [a, b]
        tensor = pybamm.TensorField([[a, b], [b, a]])
        assert tensor.components == [[a, b], [b, a]]
        assert tensor[1, 0] is b
        assert expr.orphans == expr.children
        assert all(
            orphan is child
            for orphan, child in zip(expr.orphans, expr.children, strict=True)
        )

    def test_model_tree_roundtrip(self):
        model = pybamm.lithium_ion.SPM()
        for expression in [*model.rhs.values(), *model.variables.values()][:50]:
            for node in expression.pre_order():
                _roundtrip(node)
        array = pybamm.Vector(np.ones(3), domain="negative electrode")
        rebuilt = _roundtrip(array)
        np.testing.assert_array_equal(rebuilt.entries, array.entries)


class TestTreeMapAndReplace:
    def test_replace_rederives_concatenation_scale(self):
        a = pybamm.Variable("a", domain="negative electrode")
        b = pybamm.Variable("b", domain="positive electrode")
        concatenation = pybamm.ConcatenationVariable(a, b)
        result = pybamm.replace(
            concatenation, {a: a.create_copy(scale=2), b: b.create_copy(scale=2)}
        )
        assert result.scale == pybamm.Scalar(2)
        assert all(child.scale == pybamm.Scalar(2) for child in result.children)

    def test_replace_child_and_extra_leaf_together(self):
        a = pybamm.Variable("a", domain="negative electrode")
        b = pybamm.Variable("b", domain="separator")
        x = pybamm.SpatialVariable("x", domain="negative electrode")
        y = pybamm.SpatialVariable("y", domain="separator")
        result = pybamm.replace(pybamm.Integral(a, x), {a: b, x: y})
        assert result == pybamm.Integral(b, y)

    def test_replace_moves_parent_to_the_new_child_domain(self):
        a = pybamm.Variable("a", domain="negative electrode")
        b = pybamm.Variable("b", domain="separator")
        result = pybamm.replace(pybamm.Negate(a), {a: b})
        assert result.domain == ["separator"]
        assert result == pybamm.Negate(b)
        # a plain copy keeps the node's own domains
        annotated = pybamm.Negate(a).with_domains({"primary": ["separator"]})
        assert annotated.create_copy().domain == ["separator"]

    def test_replace_rejects_none_replacement(self):
        symbol = pybamm.Variable("a")
        with pytest.raises(ValueError, match="cannot be converted"):
            pybamm.replace(symbol, {symbol: None})

    @pytest.mark.parametrize("shared_cache", [False, True])
    def test_replace_converts_only_visited_values(self, shared_cache):
        a, b = pybamm.Parameter("a"), pybamm.Parameter("b")
        expression = pybamm.Addition(a, b)
        unused = pybamm.Parameter("unused")
        mapping = {expression: 3, a: object(), unused: object()}
        cache = {} if shared_cache else None
        result = pybamm.replace(expression, mapping, cache=cache)
        assert result == pybamm.Scalar(3)
        assert mapping[expression] == 3
        if shared_cache:
            assert cache == {expression: result}

    @pytest.mark.parametrize("shared_cache", [False, True])
    def test_replace_shared_numeric_values_and_equal_keys(self, shared_cache):
        a, b = pybamm.Parameter("a"), pybamm.Parameter("b")
        shared = pybamm.Addition(a, b)
        root = pybamm.Addition(shared, pybamm.Multiplication(shared, a))
        mapping = {pybamm.Parameter("a"): 0, b: a}
        cache = {} if shared_cache else None
        replaced = pybamm.replace(root, mapping, cache=cache, simplify=False)
        result = [replaced.left, replaced.right.left, replaced.right]
        assert result[0] is result[1]
        assert result[2].left is result[0]
        assert result[0].left == pybamm.Scalar(0)
        assert result[0].right is a
        assert result[2].right is result[0].left
        if shared_cache:
            assert cache[a] is result[0].left
            assert pybamm.replace(shared, mapping, cache=cache) is result[0]

    def test_replace_cold_deep_tree(self):
        a, b = pybamm.Parameter("a"), pybamm.Parameter("b")
        expression = a
        for _ in range(3000):
            expression = pybamm.Addition(expression, a)
        assert pybamm.replace(expression, {b: 1}) is expression
        result = pybamm.replace(expression, {a: b}, simplify=False)
        for _ in range(3000):
            assert result.right is b
            result = result.left
        assert result is b

    def test_replace_shares_untouched_subtrees(self):
        a = pybamm.Variable("a", domain="negative electrode")
        b = pybamm.Variable("b", domain="negative electrode")
        c = pybamm.Variable("c", domain="negative electrode")
        untouched = pybamm.grad(b) * 3
        expr = untouched + a * pybamm.exp(a)
        result = pybamm.replace(expr, {a: c})
        assert result == untouched + c * pybamm.exp(c)
        assert result is not expr
        assert result.left is expr.left
        assert expr == untouched + a * pybamm.exp(a)  # original untouched
        # nothing to do -> the very same object
        assert pybamm.replace(expr, {b: b}) is expr
        assert pybamm.replace(expr, {}) is expr

    def test_replace_does_not_traverse_replacements(self):
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        # b -> a would loop forever if replacements were re-traversed
        result = pybamm.replace(a + b, {a: b, b: a})
        assert result == b + a

    def test_replace_numbers_and_simplification(self):
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        assert pybamm.replace(a * b, {a: 0}) == pybamm.Scalar(0)
        kept = pybamm.replace(a * b, {a: 0}, simplify=False)
        assert isinstance(kept, pybamm.Multiplication)
        assert kept.left == pybamm.Scalar(0)

    def test_replace_extra_leaf_fields(self):
        scale = pybamm.Parameter("scale")
        a = pybamm.Variable("a", domain="negative electrode", scale=scale)
        x = pybamm.SpatialVariable("x", domain="negative electrode")
        integral = pybamm.Integral(2 * a, x)
        y = pybamm.SpatialVariable("y", domain="negative electrode")
        result = pybamm.replace(integral, {scale: 3, x: y})
        assert result.integration_variable == [y]
        (variable,) = pybamm.SymbolUnpacker(pybamm.Variable).unpack_symbol(result)
        assert variable.scale == 3
        assert variable.name == "a"

    def test_cache_is_shared_and_deep_trees_work(self):
        a = pybamm.Variable("a")
        b = pybamm.Variable("b")
        expr = a
        for _ in range(3000):
            expr = expr + 1
        cache = {}
        result = pybamm.replace(expr, {a: b}, cache=cache, simplify=False)
        assert cache[a] is b
        # a second tree sharing nodes reuses the memo
        again = pybamm.replace(
            pybamm.Multiplication(expr, 2), {a: b}, cache=cache, simplify=False
        )
        assert again.left is result

    def test_tree_map_custom_fn(self):
        a = pybamm.Variable("a")
        expr = pybamm.exp(a) + pybamm.exp(a) * 2

        def double_variables(node, new_leaves):
            if isinstance(node, pybamm.Variable):
                return 2 * node
            return pybamm.rebuild(node, new_leaves)

        result = pybamm.tree_map(double_variables, expr)
        assert result == pybamm.exp(2 * a) + pybamm.exp(2 * a) * 2
        assert pybamm.tree_map(pybamm.rebuild, expr) is expr

    def test_rebuild_checks_leaf_count(self):
        a = pybamm.Variable("a")
        with pytest.raises(ValueError, match=r"expects 2 leaves, got 1"):
            pybamm.rebuild(a * 2, (a,))

    def test_replace_parameter_throughout_a_model(self):
        model = pybamm.lithium_ion.SPM()
        c_max = pybamm.Parameter(
            "Maximum concentration in negative electrode [mol.m-3]"
        )
        unpacker = pybamm.SymbolUnpacker(pybamm.Parameter)
        assert any(c_max in unpacker.unpack_symbol(eqn) for eqn in model.rhs.values())
        cache = {}
        replaced = [
            pybamm.replace(eqn, {c_max: 1234.0}, cache=cache)
            for eqn in model.rhs.values()
        ]
        assert all(c_max not in unpacker.unpack_symbol(eqn) for eqn in replaced)


class TestSymbolRoots:
    @pytest.mark.parametrize("empty_mapping", [False, True])
    def test_rejects_non_symbol_roots(self, empty_mapping):
        x = pybamm.Symbol("x")
        step = pybamm.step.current(1, duration=10)
        roots = [
            None,
            1,
            "x",
            [x],
            (x,),
            {"x": x},
            pybamm.Event("stop", x),
            pybamm.BaseModel(),
            step,
            pybamm.Experiment([step]),
        ]
        mapping = {} if empty_mapping else {x: 2}
        for root in roots:
            with pytest.raises(TypeError, match=r"tree must be a pybamm\.Symbol"):
                pybamm.replace(root, mapping)
            with pytest.raises(TypeError, match=r"tree must be a pybamm\.Symbol"):
                pybamm.tree_map(pybamm.rebuild, root)


class TestIdentityIsUniform:
    def test_no_class_overrides_identity(self):
        def subclasses(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from subclasses(sub)

        overriding = [
            cls.__qualname__
            for cls in subclasses(pybamm.Symbol)
            if "_compute_id" in cls.__dict__
        ]
        assert overriding == []

    @settings(max_examples=200, deadline=None)
    @given(symbols(max_leaves=6))
    def test_identity_is_the_layout_key(self, symbol):
        from pybamm.expression_tree.tree_util import identity_key, layout

        assert symbol.id == hash(
            identity_key(symbol, [leaf.id for leaf in symbol.leaves])
        )
        leaf_fields, state_fields, identity_fields = layout(type(symbol))
        assert leaf_fields[0] == "_children"
        assert set(identity_fields) <= set(state_fields)
        assert "_id" not in state_fields
        assert not (set(leaf_fields) & set(state_fields))

    def test_every_leaf_and_static_field_counts(self):
        a = pybamm.Variable("a", domain="negative particle size")
        f, g = pybamm.Scalar(1), pybamm.Scalar(2)
        assert pybamm.SizeAverage(a, f) != pybamm.SizeAverage(a, g)
        assert pybamm.SizeAverage(a, f) == pybamm.SizeAverage(a, pybamm.Scalar(1))
        b = pybamm.Variable("b")
        assert pybamm.EvaluateAt(b, pybamm.Scalar(0)) != pybamm.EvaluateAt(
            b, pybamm.Scalar(1)
        )
        assert pybamm.Index(pybamm.Vector([1, 2, 3]), 0) != pybamm.Index(
            pybamm.Vector([1, 2, 3]), 1
        )
        assert pybamm.Variable("v", scale=2) != pybamm.Variable("v", scale=3)
        assert pybamm.Variable("v", domain="a") != pybamm.Variable("v", domain="b")

    def test_numbers_with_equal_hashes_stay_distinct(self):
        # CPython: hash(-1) == hash(-2); identity must not inherit that collision
        assert pybamm.Scalar(-1) != pybamm.Scalar(-2)
        assert pybamm.Scalar(-1.0) != pybamm.Scalar(-2.0)
        assert pybamm.InputParameter("p", expected_size=-1) != pybamm.InputParameter(
            "p", expected_size=-2
        )
        assert pybamm.Scalar(np.float64(-1.0)) == pybamm.Scalar(-1.0)

    def test_identity_ignores_labels_annotations_and_excluded_fields(self):
        assert pybamm.Scalar(2, name="two") == pybamm.Scalar(2, name="deux")
        a = pybamm.Variable("a", domain="negative electrode")
        labelled = pybamm.Variable("a", domain="negative electrode")
        labelled.print_name = "a_n"
        assert labelled == a
        mesh_a = a.with_mesh(object())
        assert mesh_a == a
        assert pybamm.Function(abs, a) == pybamm.Function(abs, a)
        vec = pybamm.Vector([1.0, 2.0])
        assert vec == pybamm.Vector([1.0, 2.0])
        assert vec != pybamm.Vector([1.0, 3.0])
