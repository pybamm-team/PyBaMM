import pybamm

from pybamm.type_definitions import DomainType


class CoupledVariable(pybamm.Symbol):
    """
    A node in the expression tree representing a variable whose equation is set by a different model or submodel.


    Parameters
    ----------
    name : str
        name of the node
    domain : iterable of str
        list of domains that this coupled variable is valid over
    """

    def __init__(
        self,
        name: str,
        domain: DomainType = None,
    ) -> None:
        super().__init__(name, domain=domain)

    def _evaluate_for_shape(self):
        """
        Returns the scalar 'NaN' to represent the shape of a parameter.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return pybamm.evaluate_for_shape_using_domain(self.domains)

    def create_copy(self):
        """Creates a new copy of the coupled variable."""
        new_coupled_variable = CoupledVariable(self.name, self.domain)
        return new_coupled_variable

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, expr):
        self._children = expr

    def set_coupled_variable(self, symbol, expr):
        """Sets the children of the coupled variable to the expression passed in expr. If the symbol is not the coupled variable, then it searches the children of the symbol for the coupled variable. The coupled variable will be replaced by its first child (symbol.children[0], which should be expr) in the discretisation step."""
        if self == symbol:
            symbol.children = [
                expr,
            ]
        else:
            for child in symbol.children:
                self.set_coupled_variable(child, expr)
        symbol.set_id()
