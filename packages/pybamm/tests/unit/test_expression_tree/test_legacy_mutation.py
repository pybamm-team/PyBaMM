from __future__ import annotations

import ast
import copy
import functools
import heapq
import json
import operator
import os
import pickle
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

import pybamm
from pybamm.expression_tree.parameter import InputNames
from pybamm.expression_tree.symbol import (
    DomainNames,
    ReadOnlyDomains,
    SymbolChildren,
)
from pybamm.expression_tree.tree_util import _TRANSIENT_SLOTS, layout


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


# writes to attributes named like symbol slots whose receiver is not a symbol
_NON_SYMBOL_WRITES = {
    ("pybamm/spatial_methods/scikit_finite_element_3d.py", "M._shape"),
    (
        (
            "packages/pybamm/tests/unit/test_spatial_methods/"
            "test_finite_volume_unstructured.py"
        ),
        "method._mesh",
    ),
}


def _written_attributes(node: ast.AST) -> list[tuple[ast.expr, str]]:
    """The ``(receiver, attribute)`` pairs written by an assignment-like node."""
    if isinstance(node, ast.Assign | ast.Delete):
        targets = list(node.targets)
    elif isinstance(node, ast.AugAssign | ast.AnnAssign):
        targets = [node.target]
    elif (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) in ("setattr", "object.__setattr__")
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
    ):
        return [(node.args[0], node.args[1].value)]
    else:
        return []
    written = []
    while targets:
        target = targets.pop()
        if isinstance(target, ast.Tuple | ast.List):
            targets.extend(target.elts)
        elif isinstance(target, ast.Starred):
            targets.append(target.value)
        elif isinstance(target, ast.Attribute):
            written.append((target.value, target.attr))
    return written


def _is_warn_mutation(statement: ast.stmt) -> bool:
    """Whether ``statement`` is a bare ``_warn_mutation(...)`` call."""
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and getattr(
            statement.value.func, "attr", getattr(statement.value.func, "id", None)
        )
        == "_warn_mutation"
    )


def _new_instance_name(statement: ast.stmt) -> str | None:
    """The name ``statement`` binds, if it is ``name = <cls>.__new__(...)``."""
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, ast.Call)
        and getattr(statement.value.func, "attr", None) == "__new__"
    ):
        return statement.targets[0].id
    return None


class _Scope:
    """What a function has established before its current top-level statement."""

    def __init__(self, function: ast.AST):
        self.function = function
        self.warned = False
        self.created: set[str] = set()


class _SlotWriteFinder(ast.NodeVisitor):
    """
    Collect writes to symbol slots outside construction. Allowed are a symbol's
    writes to its own slots in ``__init__`` and ``__setstate__``, writes to an
    instance the function created with ``__new__``, and writes after a top-level
    ``_warn_mutation(...)`` call, which raises in the test suite.
    """

    def __init__(self, symbol_classes: set[ast.ClassDef], protected: set[str]):
        self.symbol_classes = symbol_classes
        self.protected = protected
        self.classes: list[ast.ClassDef] = []
        self.scopes: list[_Scope] = []
        self.writes: list[str] = []

    def visit_ClassDef(self, node):
        self.classes.append(node)
        self.generic_visit(node)
        self.classes.pop()

    def visit_FunctionDef(self, node):
        scope = _Scope(node)
        self.scopes.append(scope)
        for statement in node.body:
            self.visit(statement)
            # only a warning that always runs first guards the writes after it
            scope.warned = scope.warned or _is_warn_mutation(statement)
            created = _new_instance_name(statement)
            if created is not None:
                scope.created.add(created)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def generic_visit(self, node):
        for receiver, attribute in _written_attributes(node):
            if attribute in self.protected and not self._allowed(receiver):
                self.writes.append(f"{ast.unparse(receiver)}.{attribute}")
        super().generic_visit(node)

    def _allowed(self, receiver: ast.expr) -> bool:
        if not self.scopes:
            return False
        scope = self.scopes[-1]
        if scope.warned:
            return True
        if isinstance(receiver, ast.Name):
            if receiver.id in scope.created:
                return True
            if receiver.id == "self":
                if not self.classes or self.classes[-1] not in self.symbol_classes:
                    return True
                return scope.function.name in ("__init__", "__setstate__")
        return False


def _symbol_class_nodes(
    tree: ast.Module, module: str, symbol_names: set[str]
) -> set[ast.ClassDef]:
    """
    The classes ``tree`` defines that are symbols: those named in ``symbol_names``
    and those with a base that is, resolved through the file's imports and through
    the other classes it defines.
    """
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(
                (alias.asname, alias.name) for alias in node.names if alias.asname
            )
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            aliases.update(
                (alias.asname or alias.name, f"{node.module}.{alias.name}")
                for alias in node.names
            )

    def dotted(expr: ast.expr) -> str | None:
        if isinstance(expr, ast.Name):
            return aliases.get(expr.id, expr.id)
        if isinstance(expr, ast.Attribute):
            prefix = dotted(expr.value)
            return prefix and f"{prefix}.{expr.attr}"
        return None

    local: set[str] = set()

    def is_symbol(name: str | None) -> bool:
        return name is not None and (
            name in local or name in symbol_names or f"{module}.{name}" in symbol_names
        )

    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    found: set[ast.ClassDef] = set()
    while True:
        new = [
            node
            for node in classes
            if node not in found
            and (is_symbol(node.name) or any(is_symbol(dotted(b)) for b in node.bases))
        ]
        if not new:
            return found
        found.update(new)
        local.update(node.name for node in new)


def _declared_slots(node: ast.ClassDef) -> set[str]:
    """The slot names a class body declares in ``__slots__``."""
    for statement in node.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__slots__"
            for target in statement.targets
        ):
            value = statement.value
            items = value.elts if isinstance(value, ast.Tuple | ast.List) else [value]
            return {
                item.value
                for item in items
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
    return set()


@functools.cache
def _symbol_names_and_slots() -> tuple[frozenset[str], frozenset[str]]:
    """The dotted names pybamm's symbol classes are reached by, and their slots."""
    classes, stack = set(), [pybamm.Symbol]
    while stack:
        cls = stack.pop()
        classes.add(cls)
        stack.extend(cls.__subclasses__())
    names = {f"{cls.__module__}.{cls.__qualname__}" for cls in classes}
    names.update(
        f"pybamm.{cls.__name__}"
        for cls in classes
        if getattr(pybamm, cls.__name__, None) is cls
    )
    slots = set()
    for cls in classes:
        leaf_fields, state_fields, _ = layout(cls)
        slots.update(leaf_fields, state_fields)
    return frozenset(names), frozenset(slots)


_UNPROTECTED_SLOTS = _TRANSIENT_SLOTS | {"_print_name", "_raw_print_name"}


def _slot_writes(tree: ast.Module, module: str = "") -> list[str]:
    """The writes in ``tree`` to symbol slots outside construction, as
    ``receiver.attribute``. Symbol classes are resolved from the source, so the
    slots declared by subclasses defined in ``tree`` are protected too."""
    symbol_names, protected = _symbol_names_and_slots()
    symbol_nodes = _symbol_class_nodes(tree, module, symbol_names)
    for node in symbol_nodes:
        protected |= _declared_slots(node)
    finder = _SlotWriteFinder(symbol_nodes, protected - _UNPROTECTED_SLOTS)
    finder.visit(tree)
    return finder.writes


def _notebook_tree(path: Path) -> ast.Module:
    """The code cells of a notebook that parse as Python, without IPython magics."""
    body = []
    for cell in json.loads(path.read_text(encoding="utf-8")).get("cells", []):
        lines = "".join(cell.get("source", "")).splitlines()
        if cell.get("cell_type") != "code" or (lines and lines[0].startswith("%%")):
            continue
        code = "\n".join(
            line for line in lines if not line.lstrip().startswith(("%", "!"))
        )
        try:
            body.extend(ast.parse(code).body)
        except SyntaxError:
            continue
    return ast.Module(body=body, type_ignores=[])


def _pybamm_code():
    """``(path, module, tree)`` for PyBaMM's own Python: the package, then the
    tests, examples and docs of the repository, notebooks included."""
    package = Path(pybamm.__file__).parent
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package.parent).as_posix()
        module = relative.removesuffix(".py").replace("/", ".")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        yield relative, module.removesuffix(".__init__"), tree
    repository = Path(__file__).parents[5]
    for directory in ("packages/pybamm/tests", "examples", "docs"):
        for path in sorted((repository / directory).rglob("*")):
            if {"_build", ".ipynb_checkpoints"} & set(path.parts):
                continue
            if path.suffix == ".py":
                tree = ast.parse(path.read_text(encoding="utf-8"))
            elif path.suffix == ".ipynb":
                tree = _notebook_tree(path)
            else:
                continue
            yield path.relative_to(repository).as_posix(), "", tree


def _post_construction_slot_writes() -> set[tuple[str, str]]:
    """Every ``(path, receiver.attribute)`` write in PyBaMM's code that targets a
    symbol slot outside construction."""
    return {
        (path, write)
        for path, module, tree in _pybamm_code()
        for write in _slot_writes(tree, module)
    }


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

    def test_suite_forbids_construction_helper_setters(self):
        pybamm.settings.debug_mode = False
        vector = pybamm.Vector([1, 2])
        data = np.linspace(0, 1, 3)
        interpolant = pybamm.Interpolant(data, data, pybamm.t)
        state_vector = pybamm.StateVector(slice(0, 2))
        entries_string = (vector.entries_string, interpolant.entries_string)
        with pytest.raises(AttributeError, match=r"entries_string"):
            vector.entries_string = ("changed",)
        with pytest.raises(AttributeError, match=r"entries_string"):
            interpolant.entries_string = "changed"
        with pytest.raises(AttributeError, match=r"set_evaluation_array"):
            state_vector.set_evaluation_array([slice(0, 1)], None)
        assert (vector.entries_string, interpolant.entries_string) == entries_string
        assert state_vector.evaluation_array == [True, True]

    def test_accessors_return_immutable_containers(self):
        child = pybamm.Variable("a", domain="negative electrode")
        binary = pybamm.Addition(child, pybamm.Scalar(2))
        parameter = pybamm.FunctionParameter("f", {"x": child})
        # immutable sequences are returned as stored
        assert binary.children is binary.children is binary.orphans
        assert type(binary.children) is SymbolChildren
        assert child.domain is child._domains["primary"]
        assert type(child.secondary_domain) is DomainNames
        assert parameter.input_names is parameter.input_names
        assert type(parameter.input_names) is InputNames
        # the shared domains mapping is only ever handed out as a copy
        assert type(child.domains) is ReadOnlyDomains
        assert child.domains == child._domains
        assert child.domains is not child._domains
        assert child.domain == ["negative electrode"]
        assert parameter.input_names == ["x"]

    def test_immutable_sequences_behave_like_lists(self):
        child = pybamm.Variable("a", domain=["negative electrode", "separator"])
        binary = pybamm.Addition(child, pybamm.Scalar(2))
        names = child.domain
        assert names == ["negative electrode", "separator"] == names
        assert operator.ne(names, ["separator"]) is True
        assert operator.ne(names, list(names)) is False
        assert operator.add(names, ["x"]) == ["negative electrode", "separator", "x"]
        assert operator.add(["x"], names) == ["x", "negative electrode", "separator"]
        assert names * 2 == list(names) * 2
        assert repr(names) == repr(list(names)) == str(names)
        assert type(names.copy()) is list and names.copy() == names
        assert binary.children == [child, pybamm.Scalar(2)]
        assert json.loads(json.dumps(names)) == names
        for sequence in (names, binary.children, binary.children[0].domains):
            assert pickle.loads(pickle.dumps(sequence)) == sequence  # nosec B301
            assert copy.deepcopy(sequence) == sequence
            assert type(copy.copy(sequence)) is type(sequence)

    def test_base_class_methods_cannot_change_symbols(self):
        def symbols():
            first = pybamm.Variable("a", domain="negative electrode")
            second = pybamm.Variable("b", domain="negative electrode")
            binary = pybamm.Addition(first, pybamm.Scalar(2))
            parameter = pybamm.FunctionParameter("f", {"x": first})
            return first, second, binary, parameter

        edits = [
            lambda a, b, e, f: list.__setitem__(e.children, 0, pybamm.Scalar(9)),
            lambda a, b, e, f: heapq.heappush(e.children, pybamm.Scalar(9)),
            lambda a, b, e, f: list.append(a.domain, "separator"),
            lambda a, b, e, f: list.append(f.input_names, "y"),
            lambda a, b, e, f: dict.__setitem__(a.domains, "primary", ["separator"]),
            lambda a, b, e, f: dict.clear(a.to_json()["domains"]),
            lambda a, b, e, f: dict.clear(e.get_children_domains([a, b])),
            lambda a, b, e, f: dict.clear(a.read_domain_or_domains(None, None, None)),
        ]
        for edit in edits:
            first, second, binary, parameter = symbols()
            identities = [s.id for s in (first, second, binary, parameter)]
            try:
                edit(first, second, binary, parameter)
            except TypeError:
                pass  # tuples have no storage to edit
            assert [s.id for s in (first, second, binary, parameter)] == identities
            assert binary.children == [first, pybamm.Scalar(2)]
            assert first.domain == second.domain == ["negative electrode"]
            assert parameter.input_names == ["x"]
            # the interned domains every new symbol shares are untouched
            assert pybamm.Variable("c", domain="negative electrode").domain == [
                "negative electrode"
            ]
            assert pybamm.Variable("d").domain == []

    def test_container_edits_raise_and_leave_symbols_unchanged(self):
        child = pybamm.Variable("a", domain="negative electrode")
        equal = pybamm.Variable("a", domain="negative electrode")
        binary = pybamm.Addition(child, pybamm.Scalar(2))
        parameter = pybamm.FunctionParameter("f", {"x": child})
        edits = {
            r"children are immutable": [
                lambda: binary.children.__setitem__(0, child),
                lambda: binary.children.append(child),
                lambda: binary.orphans.clear(),
            ],
            r"domains are immutable": [
                lambda: child.domain.append("separator"),
                lambda: child.domain.__iadd__(["separator"]),
                lambda: child.domain.__imul__(2),
                lambda: child.secondary_domain.extend(["current collector"]),
                lambda: child.domains.update(primary=["separator"]),
                lambda: child.domains.__setitem__("primary", ["separator"]),
                lambda: child.domains.__ior__({"primary": ["separator"]}),
                lambda: child.domains.pop("primary"),
                lambda: child.domains["primary"].clear(),
            ],
            r"input names are immutable": [
                lambda: parameter.input_names.append("y"),
                lambda: parameter.input_names.__setitem__(0, "z"),
            ],
        }
        symbols = (child, equal, binary, parameter)
        identities = [symbol.id for symbol in symbols]
        for message, actions in edits.items():
            for action in actions:
                with pytest.raises(TypeError, match=message):
                    action()
        assert [symbol.id for symbol in symbols] == identities
        assert binary.children == [child, pybamm.Scalar(2)]
        assert child.domain == equal.domain == ["negative electrode"]
        assert parameter.input_names == ["x"]

    def test_unpickled_plain_lists_become_immutable(self):
        parameter = pybamm.FunctionParameter("f", {"x": pybamm.Variable("a")})
        state = parameter.__getstate__()
        state["_children"] = list(state["_children"])
        state["_input_names"] = list(state["_input_names"])
        restored = object.__new__(pybamm.FunctionParameter)
        restored.__setstate__(state)
        assert type(restored.children) is SymbolChildren
        assert type(restored.input_names) is InputNames
        assert restored == parameter

    def test_symbol_attribute_writes_are_not_intercepted(self):
        # a write hook taxes every slot a constructor sets, doubling construction
        # time; direct slot writes are rejected statically instead
        assert pybamm.Symbol.__setattr__ is object.__setattr__

    def test_pybamm_code_never_writes_symbol_slots_after_construction(self):
        assert _post_construction_slot_writes() == _NON_SYMBOL_WRITES

    def test_slot_write_finder_only_allows_construction_and_guarded_writes(self):
        # scanned like a test or example file: symbol classes come from the source
        source = textwrap.dedent(
            """
            import pybamm
            from pybamm import Scalar as ImportedScalar

            class Scalar(pybamm.Scalar):
                def __init__(self, value):
                    super().__init__(value)
                    self._value = value
                def deprecated_setter(self, value):
                    _warn_mutation("value", "")
                    self._value = value
                def writes_before_warning(self, value):
                    self._value = value
                    _warn_mutation("value", "")
                def warns_conditionally(self, value, flag):
                    if flag:
                        _warn_mutation("value", "")
                    self._value = value
                @classmethod
                def _from_json(cls, snippet, existing):
                    instance = cls.__new__(cls)
                    instance._value = snippet
                    existing._value = snippet
                    return instance
            class Subclass(Scalar):
                def bump(self):
                    self._value = 2
            class Aliased(ImportedScalar):
                __slots__ = ("_extra",)
                def bump(self):
                    self._extra = 1
            def discretise(symbol, vector):
                symbol._value = 2
                vector[symbol._value] = 1
            class Helper:
                def update(self):
                    self._value = 3
            """
        )
        assert _slot_writes(ast.parse(source)) == [
            "self._value",
            "self._value",
            "existing._value",
            "self._value",
            "self._extra",
            "symbol._value",
        ]

    def test_deprecated_setters_and_methods(self):
        run_compatibility_script(
            """
            import warnings
            import numpy as np
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
            vector = pybamm.Vector([1, 2])
            mutate(lambda: setattr(vector, "entries_string", ("changed",)))
            assert vector.entries_string == ("changed",)
            data = np.linspace(0, 1, 3)
            interpolant = pybamm.Interpolant(data, data, pybamm.t)
            mutate(lambda: setattr(interpolant, "entries_string", "changed"))
            assert interpolant.entries_string == "changed"
            state_vector = pybamm.StateVector(slice(0, 2))
            mutate(lambda: state_vector.set_evaluation_array([slice(0, 1)], None))
            assert state_vector.evaluation_array == [True]
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

    def test_domain_setter_does_not_modify_equal_symbols(self):
        run_compatibility_script(
            """
            import warnings
            import pybamm
            from pybamm.expression_tree.legacy_mutation import SymbolMutationDeprecationWarning

            warnings.simplefilter("always", SymbolMutationDeprecationWarning)
            first = pybamm.Variable("a", domain="negative electrode")
            second = pybamm.Variable("a", domain="negative electrode")
            second_id = second.id
            assert first._domains is second._domains
            with warnings.catch_warnings(record=True) as caught:
                first.domains = {"primary": ["positive electrode"]}
            assert caught
            assert first.domain == ["positive electrode"]
            assert second.domain == ["negative electrode"]
            fresh = pybamm.Variable("a", domain="negative electrode")
            assert fresh.domain == second.domain
            assert fresh.id == second_id
            """
        )

    def test_suite_forbids_child_setters(self):
        pybamm.settings.debug_mode = False
        child = pybamm.Variable("a")
        binary = pybamm.Addition(child, pybamm.Scalar(2))
        unary = pybamm.Negate(child)
        actions = [
            lambda: setattr(binary, "left", object()),
            lambda: setattr(binary, "right", object()),
            lambda: setattr(unary, "child", object()),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for action in actions:
                with pytest.raises(AttributeError):
                    action()
        assert binary.left is unary.child is child
        assert binary.right == pybamm.Scalar(2)

    def test_child_setter_compatibility(self):
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
            assert type(binary.children).__name__ == "SymbolChildren"
            assert type(unary.children).__name__ == "SymbolChildren"
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
