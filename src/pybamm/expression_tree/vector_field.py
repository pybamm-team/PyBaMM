from typing import Optional

import numpy as np

import pybamm


class VectorField3D(pybamm.Symbol):
    """
    A node in the expression tree representing a 3D vector field.
    """

    def __init__(self, x_field, y_field, z_field):
        children = [x_field, y_field, z_field]
        if not (x_field.domain == y_field.domain == z_field.domain):
            raise ValueError("All vector field components must have the same domain")

        super().__init__(
            name="vector_field_3d", children=children, domain=x_field.domain
        )
        self.x_field = x_field
        self.y_field = y_field
        self.z_field = z_field

    def create_copy(self, new_children: Optional[list[pybamm.Symbol]] = None):
        if new_children is None:
            new_children = [self.x_field, self.y_field, self.z_field]
        new_obj = VectorField3D(*new_children)
        return new_obj

    def _evaluate_for_shape(self):
        return self.children[0].evaluate_for_shape()

    @property
    def x_field(self):
        return self.children[0]

    @x_field.setter
    def x_field(self, value):
        self.children[0] = value

    @property
    def y_field(self):
        return self.children[1]

    @y_field.setter
    def y_field(self, value):
        self.children[1] = value

    @property
    def z_field(self):
        return self.children[2]

    @z_field.setter
    def z_field(self, value):
        self.children[2] = value

    def evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """
        Evaluate the vector field by concatenating x, y, z components into a column vector.
        """
        x_eval = self.x_field.evaluate(t, y, y_dot, inputs)
        y_eval = self.y_field.evaluate(t, y, y_dot, inputs)
        z_eval = self.z_field.evaluate(t, y, y_dot, inputs)

        def ensure_column_vector(arr):
            if hasattr(arr, "toarray"):
                arr = arr.toarray()
            arr = np.asarray(arr)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim == 2 and arr.shape[1] != 1:
                arr = arr.flatten().reshape(-1, 1)
            return arr

        x_col = ensure_column_vector(x_eval)
        y_col = ensure_column_vector(y_eval)
        z_col = ensure_column_vector(z_eval)

        return np.hstack([x_col, y_col, z_col])

    def evaluates_on_edges(self, dimension: str) -> bool:
        x_evaluates_on_edges = self.x_field.evaluates_on_edges(dimension)
        y_evaluates_on_edges = self.y_field.evaluates_on_edges(dimension)
        z_evaluates_on_edges = self.z_field.evaluates_on_edges(dimension)

        if x_evaluates_on_edges == y_evaluates_on_edges == z_evaluates_on_edges:
            return x_evaluates_on_edges
        else:
            raise ValueError(
                "All components must agree on whether they evaluate on edges"
            )
