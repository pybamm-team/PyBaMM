#
# Flatten / unflatten protocol for expression trees (pytree-style)
#
from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np

import pybamm

# How the leaves of one leaf field are stored on the node
_ONE = "one"  # a single symbol
_SEQ = "seq"  # a list / tuple of symbols

# Derived values that are rebuilt on demand: never stored state, never identity
_TRANSIENT_SLOTS = frozenset(
    {
        "_id",
        "_cached_shape",
        "_cached_size",
        "_saved_evaluate_for_shape",
        "_saved_evaluates_on_edges",
    }
)
# Stored state that does not take part in identity: display labels, memo caches
# and discretisation annotations
_NON_IDENTITY_SLOTS = frozenset(
    {
        "_print_name",
        "_raw_print_name",
        "_mesh",
        "_secondary_mesh",
        "_tertiary_mesh",
        "_disc_state_vector",
    }
)


@functools.cache
def layout(cls: type) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """
    The single description of how a symbol class is put together, computed once
    per class from the ``__slots__``, ``_leaf_fields`` and ``_id_excluded_fields``
    declared along its MRO.

    Returns
    -------
    tuple
        ``(leaf_fields, state_fields, identity_fields)``: the slots holding
        symbols this node depends on (children first), every other slot that is
        stored state, and the subset of those that define the node's identity.
    """
    # declarations accumulate along the MRO, base classes first
    slots: list[str] = []
    declared_leaves: list[str] = []
    excluded: set[str] = set(_NON_IDENTITY_SLOTS)
    for klass in reversed(cls.__mro__):
        declared_slots = getattr(klass, "__slots__", ())
        if isinstance(declared_slots, str):
            declared_slots = (declared_slots,)
        for slot in declared_slots:
            if slot not in slots and slot != "__weakref__":
                slots.append(slot)
        for field in klass.__dict__.get("_leaf_fields", ()):
            if field not in declared_leaves:
                declared_leaves.append(field)
        excluded.update(klass.__dict__.get("_id_excluded_fields", ()))
    leaf_fields = ("_children", *declared_leaves)
    state_fields = tuple(
        slot
        for slot in slots
        if slot not in leaf_fields and slot not in _TRANSIENT_SLOTS
    )
    identity_fields = tuple(f for f in state_fields if f not in excluded)
    return leaf_fields, state_fields, identity_fields


def _hashable(value: Any):
    """
    A hashable stand-in for a static field value. Numbers become their ``str``:
    ``hash(-1) == hash(-2)`` in CPython, so raw numbers would let distinct values
    share an id.
    """
    if type(value) is str or type(value) is bool or value is None:
        return value
    if type(value) is pybamm.expression_tree.symbol.Domains:
        return value  # interned and pre-hashed
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int | float | complex):
        return str(value)
    if isinstance(value, dict):
        return tuple((key, _hashable(item)) for key, item in value.items())
    if isinstance(value, list | tuple):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, slice):
        return (slice, value.start, value.stop, value.step)
    if isinstance(value, np.ndarray):  # hash like the equivalent nested lists
        return _hashable(value.tolist())
    return value


def leaves_of(symbol: pybamm.Symbol) -> list[pybamm.Symbol]:
    """The symbols ``symbol`` directly depends on, children first."""
    leaf_fields = layout(type(symbol))[0]
    if len(leaf_fields) == 1:
        return symbol._children
    leaves = list(symbol._children)
    for field in leaf_fields[1:]:
        value = getattr(symbol, field, None)
        if isinstance(value, pybamm.Symbol):
            leaves.append(value)
        elif isinstance(value, list | tuple) and all(
            isinstance(item, pybamm.Symbol) for item in value
        ):
            leaves.extend(value)
    return leaves


def identity_key(symbol: pybamm.Symbol, leaf_ids: Sequence[int]) -> tuple:
    """
    Everything that defines a symbol's identity: its class, its identity-bearing
    static fields and the ids of its leaves.
    """
    cls = type(symbol)
    key = [cls]
    for field in layout(cls)[2]:
        key.append(_hashable(getattr(symbol, field, None)))
    key.extend(leaf_ids)
    return tuple(key)


class TreeDef:
    """
    The static structure of a symbol: everything needed to rebuild it from its
    leaves. Returned by :func:`tree_flatten` and consumed by :func:`tree_unflatten`.

    Parameters
    ----------
    node_type : type
        The symbol class.
    n_children : int
        How many leading leaves are the node's children.
    children_container : type
        ``list`` or ``tuple``, whichever the node stores its children in.
    leaf_spec : tuple
        ``(field, kind, count, container)`` per extra leaf field, in order.
    state : dict
        The node's remaining (non-leaf, non-derived) state.
    """

    __slots__ = ("children_container", "leaf_spec", "n_children", "node_type", "state")

    def __init__(self, node_type, n_children, children_container, leaf_spec, state):
        self.node_type = node_type
        self.n_children = n_children
        self.children_container = children_container
        self.leaf_spec = leaf_spec
        self.state = state

    @property
    def num_leaves(self) -> int:
        return self.n_children + sum(count for _, _, count, _ in self.leaf_spec)

    def __repr__(self):
        fields = ", ".join(field for field, _, _, _ in self.leaf_spec)
        return (
            f"TreeDef({self.node_type.__name__}, children={self.n_children}"
            + (f", leaves=[{fields}]" if fields else "")
            + ")"
        )


def tree_flatten(symbol: pybamm.Symbol) -> tuple[tuple[pybamm.Symbol, ...], TreeDef]:
    """
    Split a symbol into its leaves (children and the other symbols it depends
    on) and a :class:`TreeDef` describing how to put it back together.
    """
    leaf_fields = layout(type(symbol))[0]
    state = symbol.__getstate__()
    leaves: list = list(state.pop("_children"))
    spec: list = []
    for field in leaf_fields[1:]:
        # a field that is unset, None or holds plain numbers stays in the state
        value = state.get(field)
        if isinstance(value, pybamm.Symbol):
            leaves.append(state.pop(field))
            spec.append((field, _ONE, 1, None))
        elif isinstance(value, list | tuple) and all(
            isinstance(item, pybamm.Symbol) for item in value
        ):
            leaves.extend(state.pop(field))
            spec.append((field, _SEQ, len(value), type(value)))
    return tuple(leaves), TreeDef(
        type(symbol), len(symbol._children), type(symbol._children), tuple(spec), state
    )


def tree_unflatten(treedef: TreeDef, leaves: Sequence[pybamm.Symbol]) -> pybamm.Symbol:
    """
    Rebuild a symbol from a :class:`TreeDef` and its leaves, without
    simplification or validation. The result has no id yet.
    """
    if len(leaves) != treedef.num_leaves:
        raise ValueError(
            f"{treedef} expects {treedef.num_leaves} leaves, got {len(leaves)}"
        )
    state = dict(treedef.state)
    state["_children"] = treedef.children_container(leaves[: treedef.n_children])
    position = treedef.n_children
    for field, kind, count, container in treedef.leaf_spec:
        if kind == _ONE:
            state[field] = leaves[position]
        else:
            state[field] = container(leaves[position : position + count])
        position += count
    new_symbol = object.__new__(treedef.node_type)
    new_symbol.__setstate__(state)
    return new_symbol


def _same(new, old) -> bool:
    """Whether a rewritten leaf is interchangeable with the original."""
    return new is old or new == old


def rebuild(
    node: pybamm.Symbol,
    new_leaves: Sequence[pybamm.Symbol],
    simplify: bool = True,
) -> pybamm.Symbol:
    """
    Return ``node`` with its leaves swapped for ``new_leaves``.

    ``node`` itself is returned when no leaf changed, so untouched subtrees are
    shared rather than copied. When only children changed the node's constructor
    is used (so derived names, domains and, if ``simplify``, simplifications are
    applied); otherwise the structure is rebuilt as-is with
    :func:`tree_unflatten`.
    """
    leaves = leaves_of(node)
    if len(new_leaves) != len(leaves):
        raise ValueError(
            f"{type(node).__name__} expects {len(leaves)} leaves, got {len(new_leaves)}"
        )
    # equal ids mean interchangeable symbols, so an equal leaf is "unchanged"
    if all(map(_same, new_leaves, leaves)):
        return node
    n_children = len(node._children)
    new_children = list(new_leaves[:n_children])
    template = node
    if not all(map(_same, new_leaves[n_children:], leaves[n_children:])):
        # install the other rewritten leaves first, so the constructor sees them all
        _, treedef = tree_flatten(node)
        template = tree_unflatten(
            treedef, [*leaves[:n_children], *new_leaves[n_children:]]
        )
    if all(map(_same, new_children, leaves[:n_children])):
        return template
    # the constructor derives names, domains and simplifications from the inputs
    return template.create_copy(
        new_children=new_children, perform_simplifications=simplify
    )


def tree_map(
    fn: Callable[[pybamm.Symbol, tuple[pybamm.Symbol, ...]], pybamm.Symbol],
    tree: pybamm.Symbol,
    cache: MutableMapping[pybamm.Symbol, pybamm.Symbol] | None = None,
    is_leaf: Callable[[pybamm.Symbol], bool] | None = None,
) -> pybamm.Symbol:
    """
    Rewrite an expression tree bottom-up, out of place.

    Parameters
    ----------
    fn : callable
        ``fn(node, new_leaves)`` returns the node to use in place of ``node``, given
        the already-rewritten leaves. Use :func:`rebuild` for the default
        "same node with new leaves" behaviour.
    tree : :class:`pybamm.Symbol`
        The root of the expression tree to rewrite.
    cache : dict, optional
        Memo of ``{original node: rewritten node}``. Shared nodes are rewritten
        once; pass the same dict across calls to share work between trees.
    is_leaf : callable, optional
        ``is_leaf(node)`` returning True stops the descent at ``node``: ``fn`` is
        then called as ``fn(node, ())`` with nothing below it rewritten.

    Returns
    -------
    The rewritten symbol. ``tree`` is not modified.
    """
    if not isinstance(tree, pybamm.Symbol):
        raise TypeError("tree must be a pybamm.Symbol")
    if cache is None:
        cache = {}

    def postprocess(node, _leaves, new_leaves):
        return fn(node, new_leaves)

    _fold(tree, cache, postprocess, is_leaf=is_leaf)
    return cache[tree]


def _fold(tree, cache, postprocess, is_leaf=None):
    """
    Iterative post-order fold for :func:`tree_map`.

    Each node is expanded once: its leaves are computed once and kept on the
    stack, then rewritten in one go when all of them are in ``cache``.
    """
    stack = [(tree, None)]
    while stack:
        node, leaves = stack[-1]
        if leaves is None:
            if node in cache:
                stack.pop()
                continue
            if is_leaf is not None and is_leaf(node):
                cache[node] = postprocess(node, (), ())
                stack.pop()
                continue
            leaves = leaves_of(node)
            # children first, left to right
            pending = [leaf for leaf in reversed(leaves) if leaf not in cache]
            if pending:
                stack[-1] = (node, leaves)
                stack.extend((leaf, None) for leaf in pending)
                continue
        stack.pop()
        new_leaves = tuple(cache[leaf] for leaf in leaves)
        cache[node] = postprocess(node, tuple(leaves), new_leaves)


def replace(
    tree: pybamm.Symbol,
    mapping: Mapping[pybamm.Symbol, Any],
    cache: MutableMapping[pybamm.Symbol, pybamm.Symbol] | None = None,
    simplify: bool = True,
) -> pybamm.Symbol:
    """
    Substitute symbols throughout an expression tree, out of place.

    Every node equal (by id) to a key of ``mapping`` is replaced by the
    corresponding value, which is not itself traversed. All other nodes are
    rebuilt with :func:`rebuild` only if something below them changed, so the
    result shares every untouched subtree with ``tree``.

    Parameters
    ----------
    tree : :class:`pybamm.Symbol`
        The root of the expression tree to substitute into.
    mapping : dict
        ``{symbol: replacement}``; replacements may be numbers.
    cache : dict, optional
        Memo shared between calls made with the same ``mapping``.
    simplify : bool, optional
        Whether rebuilt operators apply their usual simplifications (default True).
    """
    if not isinstance(tree, pybamm.Symbol):
        raise TypeError("tree must be a pybamm.Symbol")
    if not mapping:
        return tree  # nothing to substitute: the tree is returned untouched
    replacements = dict(mapping)  # values are converted to symbols when first used
    if cache is None:
        cache = {}

    def rebuild_replaced(node, leaves, new_leaves):
        if node in replacements:
            return pybamm.convert_to_symbol(replacements[node])
        return rebuild(node, new_leaves, simplify)

    _fold(tree, cache, rebuild_replaced, is_leaf=replacements.__contains__)
    return cache[tree]
