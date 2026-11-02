from __future__ import annotations

import dataclasses
import difflib
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

import pybamm
from pybamm.solvers.preconditioners.block_solvers import KLU, BlockSolver, Stage


@dataclass(frozen=True, eq=False)
class Leaf:
    """
    A block of states solved by one block solver.

    Parameters
    ----------
    indices : numpy.ndarray
        Sorted state indices of the block.
    solver : :class:`pybamm.preconditioners.BlockSolver`
        The block solver, without its stage.
    variables : tuple of str
        Names of the state variables in the block.
    """

    indices: npt.NDArray[np.int64]
    solver: BlockSolver
    variables: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class Additive:
    """
    Blocks solved independently of each other (block Jacobi).

    Parameters
    ----------
    children : tuple
        The blocks.
    """

    children: tuple[Node, ...]

    @property
    def indices(self) -> npt.NDArray[np.int64]:
        return np.sort(np.concatenate([child.indices for child in self.children]))


@dataclass(frozen=True, eq=False)
class Multiplicative:
    """
    Blocks solved in order, each after subtracting its coupling to the blocks
    before it (block Gauss-Seidel).

    Parameters
    ----------
    children : tuple
        The blocks, in the order they are solved.
    """

    children: tuple[Node, ...]

    @property
    def indices(self) -> npt.NDArray[np.int64]:
        return np.sort(np.concatenate([child.indices for child in self.children]))


@dataclass(frozen=True, eq=False)
class Schur:
    """
    Block elimination: the ``eliminated`` block is solved, and the
    ``complement`` acts on the Schur complement of the eliminated block.

    Parameters
    ----------
    eliminated, complement
        The eliminated block and the block the Schur complement acts on.
    schur_complement : str
        ``"exact"``, ``"diagonal"`` or ``"none"``; see
        :class:`pybamm.preconditioners.Stage`.
    """

    eliminated: Node
    complement: Node
    schur_complement: str = "exact"

    @property
    def indices(self) -> npt.NDArray[np.int64]:
        return np.sort(
            np.concatenate([self.eliminated.indices, self.complement.indices])
        )


Node = Leaf | Additive | Multiplicative | Schur


def compile_tree(
    model: pybamm.BaseModel, preconditioner: dict[str, BlockSolver]
) -> Node | None:
    """
    Compile the preconditioner of a discretised model into a tree of blocks.

    Unlisted state variables get :class:`KLU` in the last stage, with that stage's
    settings, so listing only the variables that need another solver is enough.
    Stages nest
    Schur eliminations in increasing order. In a stage, variables with an equal
    solver form one block, ordered by first listing.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The discretised model.
    preconditioner : dict
        Block solvers keyed by state variable name.

    Returns
    -------
    :class:`Leaf`, :class:`Additive`, :class:`Multiplicative`, :class:`Schur` or None
        The root of the tree, or None for a model without states.

    Raises
    ------
    :class:`pybamm.ModelError`
        If the model's states have no slices yet, a name is not one of its
        state variables, or a stage has conflicting settings.
    """
    if model.y_slices is None:
        raise pybamm.ModelError(
            "The preconditioner of a model can only be compiled once its states "
            "are discretised."
        )
    states = [*model.rhs, *model.algebraic]
    if not states:
        return None
    by_name = {variable.name: variable for variable in states}
    for name in preconditioner:
        if name not in by_name:
            close = difflib.get_close_matches(name, list(by_name), n=3, cutoff=0.5)
            raise pybamm.ModelError(
                f"'{name}' has a preconditioner but is not a state variable of the "
                f"model. Close matches: {close}."
            )
    solvers = {by_name[name]: solver for name, solver in preconditioner.items()}

    # Blocks are ordered by their first variable in the entries, then by state
    listed = list(solvers)
    unlisted = sorted(
        (variable for variable in states if variable not in solvers),
        key=lambda variable: model.y_slices[variable][0].start,
    )
    groups: dict[int | None, dict[BlockSolver, list]] = {}
    settings: dict[int | None, Stage] = {}
    for variable in [*listed, *unlisted]:
        solver = solvers.get(variable, KLU())
        index = solver.stage_index
        if (
            isinstance(solver.stage, Stage)
            and settings.setdefault(index, solver.stage) != solver.stage
        ):
            raise pybamm.ModelError(
                f"Preconditioner stage {index} has conflicting settings: "
                f"{settings[index]} and {solver.stage}."
            )
        unstaged = dataclasses.replace(solver, stage=None)
        groups.setdefault(index, {}).setdefault(unstaged, []).append(variable)
    staged = sorted(stage for stage in groups if stage is not None)
    # Unstaged blocks form the last stage
    order: list[int | None] = [*staged, None] if None in groups else [*staged]

    def stage_node(index):
        leaves = [
            Leaf(
                indices=_state_indices(model, variables),
                solver=solver,
                variables=tuple(variable.name for variable in variables),
            )
            for solver, variables in groups[index].items()
        ]
        if len(leaves) == 1:
            return leaves[0]
        if settings.get(index, Stage(index)).composition == "multiplicative":
            return Multiplicative(tuple(leaves))
        return Additive(tuple(leaves))

    node = stage_node(order[-1])
    for index in reversed(order[:-1]):
        schur_complement = settings.get(index, Stage(index)).schur_complement
        node = Schur(stage_node(index), node, schur_complement)
    return node


def tree_to_config(node: Node) -> dict[str, Any]:
    """
    Serialise a compiled tree to a JSON-compatible dict.

    Parameters
    ----------
    node : :class:`Leaf`, :class:`Additive`, :class:`Multiplicative` or :class:`Schur`
        Root of the tree.

    Returns
    -------
    dict
        The tree, with each block's indices as runs ``[start, stop)``.
    """
    if isinstance(node, Leaf):
        indices = np.asarray(node.indices, dtype=np.int64)
        # Blocks are mostly contiguous, so runs keep a million-state tree small
        breaks = np.flatnonzero(np.diff(indices) != 1) + 1
        runs = [
            [int(run[0]), int(run[-1]) + 1]
            for run in np.split(indices, breaks)
            if len(run)
        ]
        return {
            "type": "leaf",
            "runs": runs,
            "solver": node.solver.to_config(),
            "variables": list(node.variables),
        }
    if isinstance(node, Schur):
        return {
            "type": "schur",
            "eliminated": tree_to_config(node.eliminated),
            "complement": tree_to_config(node.complement),
            "schur_complement": node.schur_complement,
        }
    return {
        "type": "additive" if isinstance(node, Additive) else "multiplicative",
        "children": [tree_to_config(child) for child in node.children],
    }


def tree_from_config(config: dict[str, Any]) -> Node:
    """
    Rebuild a compiled tree from :func:`tree_to_config`.

    Parameters
    ----------
    config : dict
        The serialised tree.

    Returns
    -------
    :class:`Leaf`, :class:`Additive`, :class:`Multiplicative` or :class:`Schur`
        Root of the tree.
    """
    kind = config["type"]
    if kind == "leaf":
        runs = config["runs"]
        indices = (
            np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in runs])
            if runs
            else np.zeros(0, dtype=np.int64)
        )
        return Leaf(
            indices,
            BlockSolver.from_config(config["solver"]),
            tuple(config["variables"]),
        )
    if kind == "schur":
        return Schur(
            tree_from_config(config["eliminated"]),
            tree_from_config(config["complement"]),
            config["schur_complement"],
        )
    if kind in ("additive", "multiplicative"):
        children = tuple(tree_from_config(child) for child in config["children"])
        return Additive(children) if kind == "additive" else Multiplicative(children)
    raise pybamm.OptionError(f"Unknown preconditioner tree node type '{kind}'.")


def specification(node: Node) -> dict[str, Any]:
    """
    The tree as nested dictionaries of state indices and block solver settings,
    for a preconditioner backend.

    Parameters
    ----------
    node : :class:`Leaf`, :class:`Additive`, :class:`Multiplicative` or :class:`Schur`
        Root of the tree.

    Returns
    -------
    dict
    """
    if isinstance(node, Leaf):
        return {
            "type": "leaf",
            "indices": node.indices.astype(np.int64),
            **node.solver.specification(),
        }
    if isinstance(node, Schur):
        return {
            "type": "schur",
            "schur_complement": node.schur_complement,
            "children": [
                specification(node.eliminated),
                specification(node.complement),
            ],
        }
    return {
        "type": "multiplicative" if isinstance(node, Multiplicative) else "additive",
        "children": [specification(child) for child in node.children],
    }


def _state_indices(model, variables) -> npt.NDArray[np.int64]:
    return np.sort(
        np.concatenate(
            [
                np.arange(y_slice.start, y_slice.stop)
                for variable in variables
                for y_slice in model.y_slices[variable]
            ]
        )
    ).astype(np.int64)


def describe(node: Node, indent: str = "") -> str:
    """
    Human-readable outline of a tree.

    Parameters
    ----------
    node : :class:`Leaf`, :class:`Additive`, :class:`Multiplicative` or :class:`Schur`
        Root of the tree.

    Returns
    -------
    str
    """
    if isinstance(node, Leaf):
        names = ", ".join(node.variables)
        return f"{indent}{node.solver} on {len(node.indices)} states: {names}"
    if isinstance(node, Additive | Multiplicative):
        kind = "additive" if isinstance(node, Additive) else "multiplicative"
        lines = [f"{indent}{kind}"]
        lines += [describe(child, indent + "  ") for child in node.children]
        return "\n".join(lines)
    return "\n".join(
        [
            f"{indent}schur ({node.schur_complement} complement)",
            f"{indent}  eliminate:",
            describe(node.eliminated, indent + "    "),
            f"{indent}  then, on the Schur complement:",
            describe(node.complement, indent + "    "),
        ]
    )
