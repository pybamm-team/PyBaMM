from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings

import pytest

import pybamm


def run_compatibility_script(source):
    """Exercise deprecated mutation outside the strictly immutable test process."""
    environment = os.environ.copy()
    environment.pop("PYBAMM_TEST_FORBID_SYMBOL_MUTATION", None)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class TestLegacyMutation:
    @pytest.mark.parametrize(
        "action",
        [
            lambda symbol: setattr(symbol, "name", "changed"),
            lambda symbol: setattr(symbol, "domains", {"invalid": []}),
            lambda symbol: setattr(symbol, "scale", object()),
            lambda symbol: setattr(symbol, "reference", object()),
            lambda symbol: setattr(symbol, "bounds", object()),
            lambda symbol: setattr(symbol, "mesh", object()),
            lambda symbol: setattr(symbol, "secondary_mesh", object()),
            lambda symbol: setattr(symbol, "tertiary_mesh", object()),
            lambda symbol: symbol.clear_domains(),
            lambda symbol: symbol.copy_domains(pybamm.Symbol("other")),
            lambda symbol: symbol.set_id(),
            lambda symbol: symbol.domain.append("separator"),
            lambda symbol: symbol.domains.update(primary=["separator"]),
            lambda symbol: symbol.domains.__setitem__("primary", ["separator"]),
            lambda symbol: symbol.domains["primary"].clear(),
        ],
    )
    def test_suite_forbids_mutation_even_without_debug_or_warnings(self, action):
        symbol = pybamm.Variable("a", domain="negative electrode")
        identity = symbol.id
        pybamm.settings.debug_mode = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(AttributeError):
                action(symbol)
        assert symbol.id == identity
        assert symbol.name == "a"
        assert symbol.domain == ["negative electrode"]

    def test_suite_forbids_subclass_setters(self):
        pybamm.settings.debug_mode = False
        scalar = pybamm.Scalar(1)
        parameter = pybamm.FunctionParameter("f", {"x": scalar})
        with pytest.raises(AttributeError):
            scalar.value = 2
        with pytest.raises(AttributeError):
            parameter.input_names = ["changed"]
        assert scalar.value == 1
        assert parameter.input_names == ["x"]

    def test_deprecated_setters_and_methods(self):
        run_compatibility_script(
            """
            import warnings
            import pybamm
            from pybamm.expression_tree.legacy_mutation import SymbolMutationDeprecationWarning

            def mutate(action):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", SymbolMutationDeprecationWarning)
                    action()
                assert caught
                assert all(issubclass(w.category, SymbolMutationDeprecationWarning) for w in caught)
                assert all("deprecat" in str(w.message).lower() for w in caught)
                assert all(any(word in str(w.message) for word in
                               ("create_copy", "replace", "with_domains", "with_mesh"))
                           for w in caught)

            symbol = pybamm.Variable("a", domain="negative electrode")
            mutate(lambda: setattr(symbol, "name", "b"))
            assert symbol.name == "b"
            mutate(lambda: setattr(symbol, "scale", 2))
            assert symbol.scale == pybamm.Scalar(2)
            mutate(lambda: setattr(symbol, "reference", 3))
            assert symbol.reference == pybamm.Scalar(3)
            mutate(lambda: setattr(symbol, "bounds", (0, 5)))
            assert symbol.bounds == (pybamm.Scalar(0), pybamm.Scalar(5))
            concatenation = pybamm.ConcatenationVariable(
                pybamm.Variable("c_n", domain="negative electrode"),
                pybamm.Variable("c_p", domain="positive electrode"),
            )
            mutate(lambda: setattr(concatenation, "bounds", (0, 1)))
            assert concatenation.bounds == (pybamm.Scalar(0), pybamm.Scalar(1))
            for attribute in ("mesh", "secondary_mesh", "tertiary_mesh"):
                mesh = object()
                mutate(lambda: setattr(symbol, attribute, mesh))
                assert getattr(symbol, attribute) is mesh
            mutate(lambda: setattr(symbol, "domains", {"primary": ["separator"]}))
            assert symbol.domain == ["separator"]
            mutate(symbol.clear_domains)
            assert symbol.domain == []
            donor = pybamm.Variable("donor", domain="positive electrode")
            mutate(lambda: symbol.copy_domains(donor))
            assert symbol.domains == donor.domains
            identity = symbol.id
            mutate(symbol.set_id)
            assert symbol.id == identity
            scalar = pybamm.Scalar(1)
            mutate(lambda: setattr(scalar, "value", 4))
            assert scalar.evaluate() == 4
            parameter = pybamm.FunctionParameter("f", {"x": scalar})
            mutate(lambda: setattr(parameter, "input_names", ["y"]))
            assert parameter.input_names == ["y"]
            """
        )

    def test_mutation_invalidates_shared_parents_and_shape_caches(self):
        run_compatibility_script(
            """
            import warnings
            import pybamm
            from pybamm.expression_tree.legacy_mutation import SymbolMutationDeprecationWarning

            warnings.simplefilter("always", SymbolMutationDeprecationWarning)
            child = pybamm.Variable("a", domain="negative electrode")
            left = pybamm.Negate(child)
            right = pybamm.AbsoluteValue(child)
            root = pybamm.Addition(left, right)
            old_ids = [node.id for node in (child, left, right, root)]
            with warnings.catch_warnings(record=True) as caught:
                child.name = "b"
            assert caught
            assert all(node.id != old for node, old in
                       zip((child, left, right, root), old_ids, strict=True))
            assert left == pybamm.Negate(child)
            assert right == pybamm.AbsoluteValue(child)
            assert root == pybamm.Addition(left, right)
            assert left.child is right.child is child
            assert child.shape_for_testing == (11, 1)
            assert left.shape_for_testing == (11, 1)
            with warnings.catch_warnings(record=True) as caught:
                child.domains = {"primary": ["separator"]}
            assert caught
            assert child.shape_for_testing == (13, 1)
            assert left.shape_for_testing == (13, 1)
            """
        )

    def test_nested_domain_mutation_does_not_modify_equal_symbols(self):
        run_compatibility_script(
            """
            import warnings
            import pybamm
            from pybamm.expression_tree.legacy_mutation import SymbolMutationDeprecationWarning

            warnings.simplefilter("always", SymbolMutationDeprecationWarning)
            first = pybamm.Variable("a", domain="negative electrode")
            second = pybamm.Variable("a", domain="negative electrode")
            second_id = second.id
            alias = first.domain
            with warnings.catch_warnings(record=True) as caught:
                alias.append("separator")
            assert caught
            assert first.domain == ["negative electrode", "separator"]
            assert second.domain == ["negative electrode"]
            assert second.id == second_id
            with warnings.catch_warnings(record=True) as caught:
                first.domains["primary"] = ["positive electrode"]
            assert caught
            assert first.domain == ["positive electrode"]
            assert second.domain == ["negative electrode"]
            fresh = pybamm.Variable("a", domain="negative electrode")
            assert fresh.domain == second.domain
            assert fresh.id == second_id
            """
        )

    def test_suite_forbids_child_and_input_list_mutation(self):
        pybamm.settings.debug_mode = False
        child = pybamm.Variable("a")
        binary = pybamm.Addition(child, pybamm.Scalar(2))
        unary = pybamm.Negate(child)
        parameter = pybamm.FunctionParameter("f", {"x": child})
        actions = [
            lambda: setattr(binary, "left", object()),
            lambda: setattr(binary, "right", object()),
            lambda: setattr(unary, "child", object()),
            lambda: binary.children.__setitem__(0, child),
            lambda: binary.orphans.clear(),
            lambda: parameter.input_names.append("y"),
            lambda: child.domains.__ior__({"primary": ["separator"]}),
            lambda: child.domain.__iadd__(["separator"]),
            lambda: child.domain.__imul__(2),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for action in actions:
                with pytest.raises(AttributeError):
                    action()
        assert binary.left is unary.child is child
        assert binary.right == pybamm.Scalar(2)
        assert parameter.input_names == ["x"]

    def test_child_and_input_list_compatibility(self):
        run_compatibility_script(
            """
            import warnings
            import pybamm
            from pybamm.expression_tree.legacy_mutation import SymbolMutationDeprecationWarning

            warnings.simplefilter("always", SymbolMutationDeprecationWarning)
            child = pybamm.Variable("a")
            replacement = pybamm.Variable("b")
            binary = pybamm.Addition(child, pybamm.Scalar(2))
            unary = pybamm.Negate(child)
            parent = pybamm.Negate(binary)
            old_id = parent.id
            with warnings.catch_warnings(record=True) as caught:
                binary.left = replacement
                binary.right = 3
                unary.child = replacement
            assert len(caught) == 3
            assert binary.left is unary.child is replacement
            assert binary.right == pybamm.Scalar(3)
            assert parent.id != old_id
            assert parent == pybamm.Negate(pybamm.Addition(replacement, pybamm.Scalar(3)))
            children = binary.children
            orphans = binary.orphans
            with warnings.catch_warnings(record=True) as caught:
                children[0] = child
                orphans[1] = pybamm.Scalar(4)
            assert len(caught) == 2
            assert binary.left is child
            assert binary.right == pybamm.Scalar(4)
            assert children == orphans == binary.children
            parameter = pybamm.FunctionParameter("f", {"x": child})
            names = parameter.input_names
            with warnings.catch_warnings(record=True) as caught:
                parameter.input_names += ["y"]
                names[0] = "z"
                parameter.input_names *= 2
            assert caught
            assert parameter.input_names == ["z", "y", "z", "y"]
            """
        )

    def test_retained_domain_views_and_mutation_return_values(self):
        run_compatibility_script(
            """
            import warnings
            import pybamm
            from pybamm.expression_tree.legacy_mutation import SymbolMutationDeprecationWarning

            warnings.simplefilter("always", SymbolMutationDeprecationWarning)
            symbol = pybamm.Variable("a", domain="negative electrode")
            mapping = symbol.domains
            first = symbol.domain
            second = symbol.domain
            with warnings.catch_warnings(record=True) as caught:
                first.append("separator")
                second.append("positive electrode")
                mapping["secondary"] = ["current collector"]
            assert len(caught) == 3
            expected = ["negative electrode", "separator", "positive electrode"]
            assert first == second == symbol.domain == expected
            assert mapping["primary"] == expected
            assert symbol.domains["secondary"] == ["current collector"]
            with warnings.catch_warnings(record=True) as caught:
                symbol.domains |= {"tertiary": ["negative particle"]}
                symbol.domains["primary"] += ["positive particle"]
                symbol.domains["primary"] *= 1
                removed = first.pop()
                existing = mapping.setdefault("secondary", ["unused"])
            assert caught
            assert removed == "positive particle"
            assert existing == ["current collector"]
            assert symbol.domain == expected
            assert mapping["tertiary"] == ["negative particle"]
            with warnings.catch_warnings(record=True) as caught:
                removed = mapping.pop("tertiary")
                inserted = mapping.setdefault("tertiary", ["positive particle"])
                missing = mapping.pop("missing", "default")
                last_key, last_value = mapping.popitem()
            assert len(caught) == 4
            assert removed == ["negative particle"]
            assert inserted == ["positive particle"]
            assert missing == "default"
            assert last_key == "tertiary"
            assert last_value == ["positive particle"]
            assert "tertiary" not in symbol.domains
            with warnings.catch_warnings(record=True) as caught:
                symbol.domains.setdefault("tertiary", []).append("negative particle")
            assert len(caught) == 2
            assert symbol.domains["tertiary"] == ["negative particle"]
            """
        )

    def test_domain_collection_mutators(self):
        run_compatibility_script(
            """
            import warnings
            import pybamm
            from pybamm.expression_tree.legacy_mutation import SymbolMutationDeprecationWarning

            warnings.simplefilter("always", SymbolMutationDeprecationWarning)
            cases = [
                (lambda value: value.__setitem__(0, "c"), ["c", "a"], None),
                (lambda value: value.__delitem__(0), ["a"], None),
                (lambda value: value.extend(["c"]), ["b", "a", "c"], None),
                (lambda value: value.insert(1, "c"), ["b", "c", "a"], None),
                (lambda value: value.remove("b"), ["a"], None),
                (lambda value: value.sort(), ["a", "b"], None),
                (lambda value: value.reverse(), ["a", "b"], None),
                (lambda value: value.clear(), [], None),
                (lambda value: value.pop(0), ["a"], "b"),
            ]
            for mutate, expected, returned in cases:
                symbol = pybamm.Variable("a", domain=["b", "a"])
                with warnings.catch_warnings(record=True) as caught:
                    result = mutate(symbol.domain)
                assert len(caught) == 1
                assert result == returned
                assert symbol.domain == expected
            symbol = pybamm.Variable("a", domain="negative electrode")
            with warnings.catch_warnings(record=True) as caught:
                del symbol.domains["quaternary"]
            assert len(caught) == 1
            assert "quaternary" not in symbol.domains
            with warnings.catch_warnings(record=True) as caught:
                result = symbol.domains.clear()
            assert len(caught) == 1
            assert result is None
            assert symbol.domains == {}
            """
        )

    def test_legacy_domain_reordering_preserves_hash_and_symbol_identity(self):
        run_compatibility_script(
            """
            import warnings
            import pybamm

            first = pybamm.Variable(
                "a",
                domain="negative electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            second = pybamm.Variable(
                "a",
                domain="negative electrode",
                auxiliary_domains={"secondary": "current collector"},
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                primary = first.domains.pop("primary")
                first.domains["primary"] = primary
            assert first.domains == second.domains
            assert hash(first._domains) == hash(second._domains)
            assert first.id == second.id
            """
        )

    def test_child_process_inherits_strict_enforcement(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import warnings
                    import pybamm
                    pybamm.settings.debug_mode = False
                    warnings.simplefilter("ignore")
                    symbol = pybamm.Variable("a")
                    try:
                        symbol.name = "changed"
                    except AttributeError:
                        assert symbol.name == "a"
                    else:
                        raise AssertionError("Test subprocess allowed symbol mutation")
                    """
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_frozen_gc_cannot_leave_stale_symbol_ids(self):
        run_compatibility_script(
            """
            import gc
            import pybamm
            child = pybamm.Variable("x")
            parent = -child
            identity = parent.id
            gc.freeze()
            try:
                try:
                    child.name = "y"
                except RuntimeError as error:
                    assert "gc.freeze" in str(error)
                else:
                    raise AssertionError("mutation with frozen ancestors was accepted")
                assert child.name == "x"
                assert parent.id == identity
            finally:
                gc.unfreeze()
            """
        )
