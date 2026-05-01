import pybamm
from pybamm.type_definitions import DomainType


class CoupledVariable(pybamm.Symbol):
    """
    A node in the expression tree representing a variable whose equation is set by
    a different model or submodel.

    CoupledVariables are resolved lazily during discretisation by looking up the
    variable name in model.variables. This means you don't need to manually link
    coupled variables - just ensure the variable with the matching name exists in
    the combined model's variables dictionary.

    Parameters
    ----------
    name : str
        Name of the variable to couple to. Must match a key in model.variables.
    domain : iterable of str, optional
        List of domains that this coupled variable is valid over.

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
        return CoupledVariable(self.name, self.domain)
