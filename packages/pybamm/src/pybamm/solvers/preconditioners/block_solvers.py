from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import pybamm

COMPOSITIONS = ("additive", "multiplicative")
SCHUR_COMPLEMENTS = ("exact", "diagonal", "none")


@dataclass(frozen=True)
class Stage:
    """
    A stage of a preconditioner, with settings for how it is solved.

    Parameters
    ----------
    index : int, optional
        Position of the stage. Stages are eliminated by Schur complement in
        increasing order, and the last stage is solved on the remaining Schur
        complement. ``None`` (the default) is the last stage.
    composition : str, optional
        How the blocks of the stage are combined: ``"additive"`` (block Jacobi,
        the default) or ``"multiplicative"`` (block Gauss-Seidel, in the order
        their variables are listed in the preconditioner, then state order).
    schur_complement : str, optional
        How eliminating this stage forms the Schur complement of the later
        stages: ``"exact"`` (the default, using the stage's block solvers),
        ``"diagonal"`` (using the diagonal of the stage's block) or ``"none"``
        (keeping only the later stages' own block).
    """

    index: int | None = None
    composition: str = "additive"
    schur_complement: str = "exact"

    def __post_init__(self):
        if self.index is not None and (
            isinstance(self.index, bool) or not isinstance(self.index, int)
        ):
            raise pybamm.OptionError(
                f"A preconditioner stage must be an integer, not {self.index!r}."
            )
        if self.composition not in COMPOSITIONS:
            raise pybamm.OptionError(
                f"Unknown stage composition '{self.composition}'. Use one of "
                f"{', '.join(repr(c) for c in COMPOSITIONS)}."
            )
        if self.schur_complement not in SCHUR_COMPLEMENTS:
            raise pybamm.OptionError(
                f"Unknown Schur complement '{self.schur_complement}'. Use one of "
                f"{', '.join(repr(c) for c in SCHUR_COMPLEMENTS)}."
            )


@dataclass(frozen=True)
class BlockSolver:
    """
    Base class for the solver applied to one block of a preconditioner.

    A preconditioner maps state variable names to block solvers (see
    :attr:`pybamm.BaseModel.default_preconditioner`). Variables in the same stage
    with an equal solver form one block.

    Parameters
    ----------
    stage : int or :class:`Stage`, optional
        The stage of the variable; an integer is a :class:`Stage` with that
        index and default settings. ``None`` (the default) is the last stage.
    """

    stage: int | Stage | None = None

    def __post_init__(self):
        if not isinstance(self.stage, Stage):
            Stage(self.stage)

    @property
    def stage_index(self) -> int | None:
        """Index of the variable's stage."""
        return self.stage.index if isinstance(self.stage, Stage) else self.stage

    def specification(self) -> dict[str, Any]:
        """
        The solver's settings for the block preconditioner backend.

        Built-in solvers name a backend kernel, e.g. ``{"solver": "ilu", "fill":
        1}``. A solver implemented as a plugin returns ``{"solver": "external",
        "capsule": capsule, "options": options}``, where ``capsule`` is a Python
        capsule named ``"pybamm_block_solver"`` holding the C interface in
        ``pybammsolvers.get_include()/pybamm_block_solver.h``, and ``options`` a
        JSON string passed to the plugin's ``create``.

        Returns
        -------
        dict
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define its specification."
        )

    def to_json(self) -> dict[str, Any]:
        """The solver's fields, for :meth:`to_config`."""
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        if isinstance(self.stage, Stage):
            fields["stage"] = dataclasses.asdict(self.stage)
        return fields

    @classmethod
    def _from_json(cls, snippet: dict[str, Any]) -> BlockSolver:
        fields = {f.name for f in dataclasses.fields(cls)}
        values = {key: value for key, value in snippet.items() if key in fields}
        if isinstance(values.get("stage"), dict):
            values["stage"] = Stage(**values["stage"])
        return cls(**values)

    def to_config(self) -> dict[str, Any]:
        """Serialise the solver, including its class, to a JSON-compatible dict."""
        from pybamm.expression_tree.operations.serialise_kernel import encode

        return encode(self)

    @staticmethod
    def from_config(config: dict[str, Any]) -> BlockSolver:
        """
        Rebuild a solver from :meth:`to_config`. The module defining its class
        must be importable.
        """
        from pybamm.expression_tree.operations.serialise_kernel import decode

        solver = decode(config)
        if not isinstance(solver, BlockSolver):
            raise pybamm.OptionError(f"{config!r} is not a block solver configuration.")
        return solver


@dataclass(frozen=True)
class KLU(BlockSolver):
    """Sparse direct LU factorisation of the block with KLU (BTF and AMD ordering)."""

    def specification(self) -> dict[str, Any]:
        return {"solver": "klu"}


@dataclass(frozen=True)
class ILU(BlockSolver):
    """
    Incomplete LU factorisation of the block.

    Parameters
    ----------
    fill : int, optional
        Level of fill, k in ILU(k). Default is 0.
    stage : int, optional
        See :class:`BlockSolver`.
    """

    fill: int = 0

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.fill, bool) or not isinstance(self.fill, int):
            raise pybamm.OptionError(
                f"The ILU fill level must be an integer, not {self.fill!r}."
            )
        if self.fill < 0:
            raise pybamm.OptionError("The ILU fill level must be non-negative.")

    def specification(self) -> dict[str, Any]:
        return {"solver": "ilu", "fill": self.fill}


@dataclass(frozen=True)
class Jacobi(BlockSolver):
    """Inverse of the diagonal of the block."""

    def specification(self) -> dict[str, Any]:
        return {"solver": "jacobi"}


@dataclass(frozen=True)
class Tridiagonal(BlockSolver):
    """
    Inverse of the tridiagonal part of the block, in the order of its states.
    """

    def specification(self) -> dict[str, Any]:
        return {"solver": "tridiagonal"}
