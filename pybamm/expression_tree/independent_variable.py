#
# IndependentVariable class
#
import pybamm

KNOWN_COORD_SYS = ["cartesian", "spherical polar"]


class IndependentVariable(pybamm.Symbol):
    """A node in the expression tree representing an independent variable

    Used for expressing functions depending on a spatial variable or time

    Parameters
    ----------
    name : str
        name of the node
    domain : iterable of str
        list of domains that this variable is valid over

    *Extends:* :class:`Symbol`
    """

    def __init__(self, name, domain=None, auxiliary_domains=None):
        super().__init__(name, domain=domain, auxiliary_domains=auxiliary_domains)

    def _evaluate_for_shape(self):
        """ See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()` """
        return pybamm.evaluate_for_shape_using_domain(
            self.domain, self.auxiliary_domains
        )

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
        return pybamm.Scalar(0)


class Time(IndependentVariable):
    """A node in the expression tree representing time

    *Extends:* :class:`Symbol`
    """

    def __init__(self):
        super().__init__("time")

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return Time()

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """ See :meth:`pybamm.Symbol._base_evaluate()`. """
        if t is None:
            raise ValueError("t must be provided")
        return t

    def _evaluate_for_shape(self):
        """
        Return the scalar '0' to represent the shape of the independent variable `Time`.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return 0


class SpatialVariable(IndependentVariable):
    """A node in the expression tree representing a spatial variable

    Parameters
    ----------
    name : str
        name of the node (e.g. "x", "y", "z", "r", "x_n", "x_s", "x_p", "r_n", "r_p")
    domain : iterable of str
        list of domains that this variable is valid over (e.g. "cartesian", "spherical
        polar")

    *Extends:* :class:`Symbol`
    """

    def __init__(self, name, domain=None, auxiliary_domains=None, coord_sys=None):
        self.coord_sys = coord_sys
        super().__init__(name, domain=domain, auxiliary_domains=auxiliary_domains)
        domain = self.domain

        if domain == []:
            raise ValueError("domain must be provided")

        # Check symbol name vs domain name
        if name == "r" and not (len(domain) == 1 and "particle" in domain[0]):
            raise pybamm.DomainError("domain must be particle if name is 'r'")
        elif name == "r_n" and domain != ["negative particle"]:
            raise pybamm.DomainError(
                "domain must be negative particle if name is 'r_n'"
            )
        elif name == "r_p" and domain != ["positive particle"]:
            raise pybamm.DomainError(
                "domain must be positive particle if name is 'r_p'"
            )
        elif name in ["x", "y", "z", "x_n", "x_s", "x_p"] and any(
            ["particle" in dom for dom in domain]
        ):
            raise pybamm.DomainError(
                "domain cannot be particle if name is '{}'".format(name)
            )

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return self.__class__(
            self.name, self.domain, self.auxiliary_domains, self.coord_sys
        )


class SpatialVariableEdge(SpatialVariable):
    """A node in the expression tree representing a spatial variable, which evaluates
    on the edges

    Parameters
    ----------
    name : str
        name of the node (e.g. "x", "y", "z", "r", "x_n", "x_s", "x_p", "r_n", "r_p")
    domain : iterable of str
        list of domains that this variable is valid over (e.g. "cartesian", "spherical
        polar")

    *Extends:* :class:`Symbol`
    """

    def __init__(self, name, domain=None, auxiliary_domains=None, coord_sys=None):
        super().__init__(name, domain, auxiliary_domains, coord_sys)

    def evaluates_on_edges(self):
        return True


# the independent variable time
t = Time()
