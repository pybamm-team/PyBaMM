#
# Parameter classes
#
import numpy as np
import pybamm


class InputParameter(pybamm.Symbol):
    """A node in the expression tree representing an input parameter

    This node's value can be set at the point of solving, allowing parameter estimation
    and control

    Parameters
    ----------
    name : str
        name of the node

    """

    def __init__(self, name):
        super().__init__(name)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return InputParameter(self.name)

    def _evaluate_for_shape(self):
        """
        Returns the scalar 'NaN' to represent the shape of a parameter.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return np.nan

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
        return pybamm.Scalar(0)

    def _base_evaluate(self, t=None, y=None, u=None):
        # u should be a dictionary
        # convert 'None' to empty dictionary for more informative error
        if u is None:
            u = {}
        if not isinstance(u, dict):
            # if the special input "shape test" is passed, just return 1
            if u == "shape test":
                return 1
            raise TypeError("inputs u should be a dictionary")
        try:
            return u[self.name]
        # raise more informative error if can't find name in dict
        except KeyError:
            raise KeyError("Input parameter '{}' not found".format(self.name))
