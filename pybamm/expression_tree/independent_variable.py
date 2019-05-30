#
# IndependentVariable class
#
import numpy as np
import pybamm

KNOWN_SPATIAL_VARS = ["x", "y", "z", "r", "x_n", "x_s", "x_p", "r_n", "r_p"]


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

    def __init__(self, name, domain=[]):
        super().__init__(name, domain=domain)

    def evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of an IndependentVariable.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return self.evaluate_for_shape_using_domain()


class Time(IndependentVariable):
    """A node in the expression tree representing time

    *Extends:* :class:`Symbol`
    """

    def __init__(self):
        super().__init__("time")

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return Time()

    def _base_evaluate(self, t, y=None):
        """ See :meth:`pybamm.Symbol._base_evaluate()`. """
        if t is None:
            raise ValueError("t must be provided")
        return t

    def evaluate_for_shape(self):
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
        name of the node (e.g. "x_n")
    domain : iterable of str
        list of domains that this variable is valid over

    *Extends:* :class:`Symbol`
    """

    def __init__(self, name, domain=[], coord_sys=None):
        self.coord_sys = coord_sys
        super().__init__(name, domain=domain)

        if name not in KNOWN_SPATIAL_VARS:
            raise ValueError(
                "name must be KNOWN_SPATIAL_VARS  but is '{}'".format(name)
            )
        if domain == []:
            raise ValueError("domain must be provided")
        if name in ["r", "r_n", "r_p"] and domain not in [
            ["negative particle"],
            ["positive particle"],
        ]:
            raise pybamm.DomainError("domain must be particle if name is 'r'")
        elif name in ["x", "y", "z", "x_n", "x_s", "x_p"] and any(
            ["particle" in dom for dom in domain]
        ):
            raise pybamm.DomainError(
                "domain cannot be particle if name is '{}'".format(name)
            )

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return SpatialVariable(self.name, self.domain, self.coord_sys)


# the independent variable time
t = Time()
