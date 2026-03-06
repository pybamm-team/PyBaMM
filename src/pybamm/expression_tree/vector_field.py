"""
VectorField class - a rank-1 tensor field with N components.
"""

from __future__ import annotations

import pybamm
from pybamm.expression_tree.tensor_field import TensorField


class VectorField(TensorField):
    """
    A node in the expression tree representing a vector field.

    VectorField is a convenience subclass of TensorField for rank-1 tensors
    with N >= 2 components.  Components are stored by integer index; the
    properties ``lr_field``, ``tb_field``, and ``fb_field`` are backward-
    compatible aliases for ``[0]``, ``[1]``, and ``[2]``.

    Parameters
    ----------
    *components : pybamm.Symbol
        Two or more component symbols, all sharing the same domain.
    """

    def __init__(self, *components):
        if len(components) < 2:
            raise ValueError(
                f"VectorField requires at least 2 components, got {len(components)}"
            )
        ref_domain = components[0].domain
        for i, c in enumerate(components[1:], start=1):
            if c.domain != ref_domain:
                raise ValueError(
                    f"All components must have the same domain: "
                    f"component {i} has {c.domain}, expected {ref_domain}"
                )
        super().__init__(list(components), domain=ref_domain)
        self.name = "vector_field"

    @property
    def n_components(self):
        """Number of vector components."""
        return len(self._components)

    # ---- backward-compatible aliases for structured-grid directions ----

    @property
    def lr_field(self):
        """Component 0 (left-right / x)."""
        return self._components[0]

    @property
    def tb_field(self):
        """Component 1 (top-bottom / y)."""
        return self._components[1]

    @property
    def fb_field(self):
        """Component 2 (front-back / z).  Only valid for 3-component fields."""
        if len(self._components) < 3:
            raise AttributeError(
                "fb_field requires at least 3 components; this VectorField has "
                f"{len(self._components)}"
            )
        return self._components[2]

    def create_copy(
        self,
        new_children: list[pybamm.Symbol] | None = None,
        perform_simplifications: bool = True,
    ):
        if new_children is None:
            new_children = [
                c.create_copy(perform_simplifications=perform_simplifications)
                for c in self._components
            ]
        return VectorField(*new_children)

    def evaluates_on_edges(self, dimension: str) -> bool:
        statuses = [c.evaluates_on_edges(dimension) for c in self._components]
        if all(statuses):
            return True
        if not any(statuses):
            return False
        raise ValueError(
            "All VectorField components must either all evaluate on edges "
            "or none evaluate on edges"
        )
