import pybamm
import numpy as np


class VectorField3D(pybamm.Symbol):
    def __init__(self, x_field, y_field, z_field):
        if not (x_field.domain == y_field.domain == z_field.domain):
            raise ValueError("All vector field components must have the same domain")

        self.x_field = x_field
        self.y_field = y_field
        self.z_field = z_field

        super().__init__(
            name="vector_field_3d",
            children=[x_field, y_field, z_field],
            domain=x_field.domain,
        )

    def create_copy(self, new_children=None):
        if new_children is None:
            new_children = [self.x_field, self.y_field, self.z_field]
        return VectorField3D(*new_children)

    def evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """Evaluate the vector field by evaluating each component"""
        x_val = self.x_field.evaluate(t, y, y_dot, inputs)
        y_val = self.y_field.evaluate(t, y, y_dot, inputs)
        z_val = self.z_field.evaluate(t, y, y_dot, inputs)
        return np.column_stack([x_val.flatten(), y_val.flatten(), z_val.flatten()])

    def _evaluate_for_shape(self):
        return self.x_field.evaluate_for_shape()

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
        # The Jacobian of a vector field is the Jacobian of each component
        jac_x = self.x_field.jac(variable)
        jac_y = self.y_field.jac(variable)
        jac_z = self.z_field.jac(variable)

        # For a vector field, we return a VectorField3D of the Jacobians
        return VectorField3D(jac_x, jac_y, jac_z)
