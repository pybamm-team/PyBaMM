#
# IndependentVariable class
#
import sympy

import pybamm

KNOWN_COORD_SYS = ["cartesian", "cylindrical polar", "spherical polar"]


class IndependentVariable(pybamm.Symbol):
    """
    A node in the expression tree representing an independent variable.

    Used for expressing functions depending on a spatial variable or time

    Parameters
    ----------
    name : str
        name of the node
    domain : iterable of str
        list of domains that this variable is valid over
    auxiliary_domains : dict, optional
        dictionary of auxiliary domains, defaults to empty dict
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    """

    def __init__(self, name, domain=None, auxiliary_domains=None, domains=None):
        super().__init__(
            name, domain=domain, auxiliary_domains=auxiliary_domains, domains=domains
        )

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        return pybamm.evaluate_for_shape_using_domain(self.domains)

    def _jac(self, variable):
        """See :meth:`pybamm.Symbol._jac()`."""
        return pybamm.Scalar(0)

    def to_equation(self):
        """Convert the node and its subtree into a SymPy equation."""
        if self.print_name is not None:
            return sympy.Symbol(self.print_name)
        else:
            return sympy.Symbol(self.name)


class Time(IndependentVariable):
    """
    A node in the expression tree representing time.
    """

    def __init__(self):
        super().__init__("time")

    def create_copy(self):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return Time()

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """See :meth:`pybamm.Symbol._base_evaluate()`."""
        if t is None:
            raise ValueError("t must be provided")
        return t

    def _evaluate_for_shape(self):
        """
        Return the scalar '0' to represent the shape of the independent variable `Time`.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return 0

    def to_equation(self):
        """Convert the node and its subtree into a SymPy equation."""
        return sympy.Symbol("t")


class SpatialVariable(IndependentVariable):
    """
    A node in the expression tree representing a spatial variable.

    Parameters
    ----------
    name : str
        name of the node (e.g. "x", "y", "z", "r", "x_n", "x_s", "x_p", "r_n", "r_p")
    domain : iterable of str
        list of domains that this variable is valid over (e.g. "cartesian", "spherical
        polar")
    auxiliary_domains : dict, optional
        dictionary of auxiliary domains, defaults to empty dict
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    """

    def __init__(
        self, name, domain=None, auxiliary_domains=None, domains=None, coord_sys=None
    ):
        self.coord_sys = coord_sys
        super().__init__(
            name, domain=domain, auxiliary_domains=auxiliary_domains, domains=domains
        )
        domain = self.domain

        if domain == []:
            raise ValueError("domain must be provided")

        # Check symbol name vs domain name
        if name == "r_n" and not all(n in domain[0] for n in ["negative", "particle"]):
            # catches "negative particle", "negative secondary particle", etc
            raise pybamm.DomainError(
                "domain must be negative particle if name is 'r_n'"
            )
        elif name == "r_p" and not all(
            n in domain[0] for n in ["positive", "particle"]
        ):
            # catches "positive particle", "positive secondary particle", etc
            raise pybamm.DomainError(
                "domain must be positive particle if name is 'r_p'"
            )
        elif name in ["x", "y", "z", "x_n", "x_s", "x_p"] and any(
            ["particle" in dom for dom in domain]
        ):
            raise pybamm.DomainError(
                "domain cannot be particle if name is '{}'".format(name)
            )

    def create_copy(self):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return self.__class__(self.name, domains=self.domains, coord_sys=self.coord_sys)


class SpatialVariableEdge(SpatialVariable):
    """
    A node in the expression tree representing a spatial variable, which evaluates
    on the edges

    Parameters
    ----------
    name : str
        name of the node (e.g. "x", "y", "z", "r", "x_n", "x_s", "x_p", "r_n", "r_p")
    domain : iterable of str
        list of domains that this variable is valid over (e.g. "cartesian", "spherical
        polar")
    auxiliary_domains : dict, optional
        dictionary of auxiliary domains, defaults to empty dict
    domains : dict
        A dictionary equivalent to {'primary': domain, auxiliary_domains}. Either
        'domain' and 'auxiliary_domains', or just 'domains', should be provided
        (not both). In future, the 'domain' and 'auxiliary_domains' arguments may be
        deprecated.
    """

    def __init__(
        self, name, domain=None, auxiliary_domains=None, domains=None, coord_sys=None
    ):
        super().__init__(name, domain, auxiliary_domains, domains, coord_sys)

    def _evaluates_on_edges(self, dimension):
        return True


# the independent variable time
t = Time()
