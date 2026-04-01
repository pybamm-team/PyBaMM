"""
VectorField class - a rank-1 tensor field for 2D simulations.
"""

from __future__ import annotations

import pybamm
from pybamm.expression_tree.tensor_field import TensorField


class VectorField(TensorField):
    """
    A node in the expression tree representing a vector field.

    VectorField is a convenience subclass of TensorField for rank-1 tensors
    with two components (lr and tb directions in 2D).

    Parameters
    ----------
    lr_field : pybamm.Symbol
        The left-right (x) component of the vector field.
    tb_field : pybamm.Symbol
        The top-bottom (y) component of the vector field.
    """

    def __init__(self, lr_field, tb_field):
        if lr_field.domain != tb_field.domain:
            raise ValueError("lr_field and tb_field must have the same domain")
        # Initialize as a rank-1 TensorField with two components
        super().__init__([lr_field, tb_field], domain=lr_field.domain)
        # Override the name to maintain backward compatibility
        self.name = "vector_field"

    @property
    def lr_field(self):
        """The left-right (x) component of the vector field."""
        return self._components[0]

    @property
    def tb_field(self):
        """The top-bottom (y) component of the vector field."""
        return self._components[1]

    def create_copy(
        self,
        new_children: list[pybamm.Symbol] | None = None,
        perform_simplifications: bool = True,
    ):
        """Create a copy of this vector field with optional new children."""
        if new_children is None:
            new_children = [
                self.lr_field.create_copy(
                    perform_simplifications=perform_simplifications
                ),
                self.tb_field.create_copy(
                    perform_simplifications=perform_simplifications
                ),
            ]
        return VectorField(*new_children)

    def evaluates_on_edges(self, dimension: str) -> bool:
        """Check if components evaluate on edges.

        Overrides TensorField to provide more specific error message.
        """
        left_evaluates_on_edges = self.lr_field.evaluates_on_edges(dimension)
        right_evaluates_on_edges = self.tb_field.evaluates_on_edges(dimension)
        if left_evaluates_on_edges == right_evaluates_on_edges:
            return left_evaluates_on_edges
        else:
            raise ValueError(
                "lr_field and tb_field must either both evaluate on edges "
                "or both not evaluate on edges"
            )
