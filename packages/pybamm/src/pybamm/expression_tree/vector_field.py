"""
VectorField class - a rank-1 tensor field with N components.
"""

from __future__ import annotations

import casadi

import pybamm
from pybamm.expression_tree.tensor_field import TensorField


class VectorField(TensorField):
    """
    A node in the expression tree representing a vector field.

    VectorField is a convenience subclass of TensorField for rank-1 tensors
    with N >= 2 components.  Components are stored by integer index; the
    properties ``lr_field`` and ``tb_field`` are aliases for ``[0]`` and ``[1]``.

    Parameters
    ----------
    *components : pybamm.Symbol
        Two or more component symbols, all sharing the same domain.
    """

    # ``disc_state_vector`` is attached by the discretisation for unstructured FV
    # edge-averaging (None until then) and is not part of the serialised form.
    __slots__ = ("_disc_state_vector",)
    _serialise_derived_params = frozenset({"disc_state_vector"})

    def __init__(
        self,
        *components: pybamm.Symbol,
        disc_state_vector: pybamm.StateVector | None = None,
    ) -> None:
        if len(components) < 2:
            raise ValueError(
                f"VectorField requires at least 2 components, got {len(components)}"
            )
        ref_domain = components[0]._domains["primary"]
        for i, c in enumerate(components[1:], start=1):
            if c._domains["primary"] != ref_domain:
                raise ValueError(
                    f"All components must have the same domain: "
                    f"component {i} has {c._domains['primary']}, expected {ref_domain}"
                )
        super().__init__(list(components), domain=ref_domain, name="vector_field")
        self._disc_state_vector = disc_state_vector

    @classmethod
    def _from_json(cls, snippet):
        # N positional args, not a single list -- override TensorField._from_json.
        return cls(*snippet["children"])

    disc_state_vector = pybamm.expression_tree.legacy_mutation.legacy_property(
        "_disc_state_vector",
        "Construct a new VectorField with disc_state_vector=value.",
        doc="State vector attached by the unstructured FV discretisation (else None).",
    )

    def with_meshes(
        self,
        mesh: pybamm.Mesh,
        domains: dict[str, list[str]] | None = None,
    ) -> VectorField:
        """See :meth:`pybamm.Symbol.with_meshes()`; components carry the meshes too."""
        domains = self._domains if domains is None else domains
        with_components = VectorField(
            *(component.with_meshes(mesh, domains) for component in self.components),
            disc_state_vector=self.disc_state_vector,
        )
        return super(VectorField, with_components).with_meshes(mesh, domains)

    @property
    def n_components(self) -> int:
        """Number of vector components."""
        return len(self.components)

    # ---- aliases for structured-grid directions ----

    @property
    def lr_field(self) -> pybamm.Symbol:
        """Component 0 (left-right / x)."""
        return self.components[0]

    @property
    def tb_field(self) -> pybamm.Symbol:
        """Component 1 (top-bottom / y)."""
        return self.components[1]

    def create_copy(
        self,
        new_children: list[pybamm.Symbol] | None = None,
        perform_simplifications: bool = True,
    ) -> VectorField:
        if new_children is None:
            new_children = [
                c.create_copy(perform_simplifications=perform_simplifications)
                for c in self.components
            ]
        return VectorField(*new_children)

    def _to_casadi(self, t, y, y_dot, inputs, casadi_symbols):
        """See :meth:`pybamm.Symbol._to_casadi()`."""
        return casadi.vertcat(
            *self._children_to_casadi(t, y, y_dot, inputs, casadi_symbols)
        )

    def evaluates_on_edges(self, dimension: str) -> bool:
        statuses = [c.evaluates_on_edges(dimension) for c in self.components]
        if all(statuses):
            return True
        if not any(statuses):
            return False
        raise ValueError(
            "All VectorField components must either all evaluate on edges "
            "or none evaluate on edges"
        )
