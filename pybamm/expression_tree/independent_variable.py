#
# IndependentVariable class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


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


class Time(IndependentVariable):
    """A node in the expression tree representing time

    *Extends:* :class:`Symbol`
    """

    def __init__(self):
        super().__init__("time")

    def evaluate(self, t, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        if t is None:
            raise ValueError("t must be provided")
        return t


class SpatialVariable(IndependentVariable):
    """A node in the expression tree representing a spatial variable

    Parameters
    ----------
    name : str
        name of the node ("x", "y", "z" or "r")
    domain : iterable of str
        list of domains that this variable is valid over

    *Extends:* :class:`Symbol`
    """

    def __init__(self, name, domain):
        if name not in ["x", "y", "z", "r"]:
            raise ValueError(
                "name must be 'x', 'y', 'z' or 'r' but is '{}'".format(name)
            )
        if domain == []:
            raise ValueError("domain must be provided")
        if name == "r" and domain not in [["negative particle"], ["positive particle"]]:
            raise pybamm.DomainError("domain must be particle if name is 'r'")
        elif name in ["x", "y", "z"] and any(["particle" in dom for dom in domain]):
            raise pybamm.DomainError(
                "domain cannot be particle if name is '{}'".format(name)
            )

        super().__init__(name, domain=domain)


# the independent variable time
t = Time()
