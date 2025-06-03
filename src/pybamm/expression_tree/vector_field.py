import pybamm
from typing import Optional
import numpy as np


class VectorField3D(pybamm.Symbol):
    def __init__(self, x_field, y_field, z_field):
        if not (x_field.domain == y_field.domain == z_field.domain):
            raise ValueError("All vector field components must have the same domain")

        super().__init__(
            name="vector_field_3d",
            children=[x_field, y_field, z_field],
            domain=x_field.domain,
        )
        self.x_field = x_field
        self.y_field = y_field
        self.z_field = z_field

    def create_copy(self, new_children: Optional[list[pybamm.Symbol]] = None):
        if new_children is None:
            new_children = [self.x_field, self.y_field, self.z_field]
        return VectorField3D(*new_children)

    def _evaluate_for_shape(self):
        return self.children[0].evaluate_for_shape()

    def evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """
        Evaluate the vector field by evaluating each component
        and returning an object with x_field, y_field, and z_field attributes.
        """
        x_eval = self.x_field.evaluate(t, y, y_dot, inputs)
        y_eval = self.y_field.evaluate(t, y, y_dot, inputs)
        z_eval = self.z_field.evaluate(t, y, y_dot, inputs)

        class VectorResult:
            def __init__(self, x, y, z):
                self.x_field = x
                self.y_field = y
                self.z_field = z

            def flatten(self):
                """Convert to a flat array for comparison"""
                x_flat = (
                    self.x_field.flatten()
                    if hasattr(self.x_field, "flatten")
                    else np.array(self.x_field).flatten()
                )
                y_flat = (
                    self.y_field.flatten()
                    if hasattr(self.y_field, "flatten")
                    else np.array(self.y_field).flatten()
                )
                z_flat = (
                    self.z_field.flatten()
                    if hasattr(self.z_field, "flatten")
                    else np.array(self.z_field).flatten()
                )
                return np.concatenate([x_flat, y_flat, z_flat])

            def __array__(self):
                """Support numpy array conversion"""
                return self.flatten()

            def __eq__(self, other):
                """Support equality comparison with arrays"""
                if isinstance(other, np.ndarray):
                    if len(other.shape) > 1 and other.shape[1] == 1:
                        # If other is a column vector, flatten it
                        other = other.flatten()
                    return np.array_equal(self.flatten(), other)
                return NotImplemented

        return VectorResult(x_eval, y_eval, z_eval)

    def evaluates_on_edges(self, dim):
        vals = [
            f.evaluates_on_edges(dim)
            for f in (self.x_field, self.y_field, self.z_field)
        ]
        if vals.count(vals[0]) == 3:
            return vals[0]
        else:
            raise ValueError("components disagree on edges")

    def _jac(self, variable):
        """Compute the Jacobian of the vector field"""
        jac_x = self.x_field.jac(variable)
        jac_y = self.y_field.jac(variable)
        jac_z = self.z_field.jac(variable)

        return VectorField3D(jac_x, jac_y, jac_z)
