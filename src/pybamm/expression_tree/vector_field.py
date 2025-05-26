import pybamm


class VectorField3D(pybamm.Symbol):
    def __init__(self, x_field, y_field, z_field):
        if not (x_field.domain == y_field.domain == z_field.domain):
            raise ValueError(...)
        super().__init__(
            name="vector_field_3d",
            children=[x_field, y_field, z_field],
            domain=x_field.domain,
        )
        self.x_field, self.y_field, self.z_field = x_field, y_field, z_field

    def create_copy(self, new_children=None):
        if new_children is None:
            new_children = [self.x_field, self.y_field, self.z_field]
        return VectorField3D(*new_children)

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
