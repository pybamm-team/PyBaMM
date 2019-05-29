#
# IndependentVariable class
#
import pybamm

KNOWN_COORD_SYS = ["cartesian", "spherical polar"]
KNOWN_SPATIAL_VARS = ["x", "y", "z", "r", "x_n", "x_s", "x_p", "r_n", "r_p"]
KNOWN_SPATIAL_VARS_EXTENDED = [v + "_edge" for v in KNOWN_SPATIAL_VARS]
KNOWN_SPATIAL_VARS.extend(KNOWN_SPATIAL_VARS_EXTENDED)


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

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return Time()

    def _base_evaluate(self, t, y=None):
        """ See :meth:`pybamm.Symbol._base_evaluate()`. """
        if t is None:
            raise ValueError("t must be provided")
        return t


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
            import ipdb

            ipdb.set_trace()
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
