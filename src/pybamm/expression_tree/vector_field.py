import pybamm


class VectorField(pybamm.Symbol):
    """
    A node in the expression tree representing a vector field.
    """

    def __init__(self, lr_field, tb_field):
        children = [lr_field, tb_field]
        if lr_field.domain != tb_field.domain:
            raise ValueError("lr_field and tb_field must have the same domain")
        self.lr_field = lr_field
        self.tb_field = tb_field
        super().__init__(name="vector_field", children=children, domain=lr_field.domain)

    def create_copy(
        self,
        new_children: list[pybamm.Symbol] | None = None,
        perform_simplifications: bool = True,
    ):
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

    def _evaluate_for_shape(self):
        return self.lr_field.evaluate_for_shape()

    def evaluates_on_edges(self, dimension: str) -> bool:
        left_evaluates_on_edges = self.lr_field.evaluates_on_edges(dimension)
        right_evaluates_on_edges = self.tb_field.evaluates_on_edges(dimension)
        if left_evaluates_on_edges:
            left_evaluates_on_edges = True
        if right_evaluates_on_edges:
            right_evaluates_on_edges = True
        if left_evaluates_on_edges == right_evaluates_on_edges:
            return left_evaluates_on_edges
        else:
            raise ValueError(
                "lr_field and tb_field must either both evaluate on edges or both not evaluate on edges"
            )
