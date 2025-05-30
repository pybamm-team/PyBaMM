import pybamm
from typing import Optional


class VectorField(pybamm.Symbol):
    """
    A node in the expression tree representing a vector field.
    """

    def __init__(self, lr_field, tb_field):
        children = [lr_field, tb_field]
        if lr_field.domain != tb_field.domain:
            raise ValueError("lr_field and tb_field must have the same domain")
        super().__init__(name="vector_field", children=children, domain=lr_field.domain)
        self.lr_field = lr_field
        self.tb_field = tb_field

    def create_copy(self, new_children: Optional[list[pybamm.Symbol]] = None):
        if new_children is None:
            new_children = [self.lr_field, self.tb_field]
        return VectorField(*new_children)

    def _evaluate_for_shape(self):
        return self.lr_field.evaluate_for_shape()

    def evaluates_on_edges(self, dimension: str) -> bool:
        left_evaluates_on_edges = self.lr_field.evaluates_on_edges(dimension)
        right_evaluates_on_edges = self.tb_field.evaluates_on_edges(dimension)
        if left_evaluates_on_edges == right_evaluates_on_edges:
            return left_evaluates_on_edges
        else:
            raise ValueError("lr_field and tb_field must have the same domain")
