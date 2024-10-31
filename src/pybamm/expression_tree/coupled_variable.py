import pybamm

from pybamm.type_definitions import DomainType


class CoupledVariable(pybamm.Symbol):
    """
    A node in the expression tree representing a variable whose equation is set by a different model or submodel.

    
    Parameters
    ----------
    name : str
        The variable's name. If the 
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
        """See :meth:`pybamm.Symbol.new_copy()`."""
        new_input_parameter = CoupledVariable(
            self.name, self.domain, expected_size=self._expected_size
        )
        return new_input_parameter

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, expr):
        self._children = expr


    def set_coupled_variable(self, symbol, expr):
        if self == symbol:
            symbol.children = [expr,]
        else:
            for child in symbol.children:
                self.set_coupled_variable(child, expr)
        symbol.set_id()