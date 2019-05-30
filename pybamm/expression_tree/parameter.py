#
# Parameter classes
#
import numpy as np
import pybamm


class Parameter(pybamm.Symbol):
    """A node in the expression tree representing a parameter

    This node will be replaced by a :class:`.Scalar` node by :class`.Parameter`

    Parameters
    ----------

    name : str
        name of the node
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list

    """

    def __init__(self, name, domain=[]):
        super().__init__(name, domain=domain)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return Parameter(self.name, self.domain)

    def evaluate_for_shape(self, t=None, y=None):
        """
        Returns the scalar 'NaN' to represent the shape of a parameter.
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        return np.nan


class FunctionParameter(pybamm.UnaryOperator):
    """A node in the expression tree representing a function parameter

    This node will be replaced by a :class:`pybamm.Function` node if a callable function
    is passed to the parameter values, and otherwise (in some rarer cases, such as
    constant current) a :class:`pybamm.Scalar` node.

    Parameters
    ----------

    name : str
        name of the node
    child : :class:`Symbol`
        child node
    diff_variable : :class:`pybamm.Symbol`, optional
        if diff_variable is specified, the FunctionParameter node will be replaced by a
        :class:`pybamm.Function` and then differentiated with respect to diff_variable.
        Default is None.

    """

    def __init__(self, name, child, diff_variable=None):
        # assign diff variable
        self.diff_variable = diff_variable
        super().__init__(name, child)

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id` """
        self._id = hash(
            (self.__class__, self.name, self.diff_variable)
            + tuple([child.id for child in self.children])
            + tuple(self.domain)
        )

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        # return a new FunctionParameter, that knows it will need to be differentiated
        # when the parameters are set
        return FunctionParameter(self.name, self.orphans[0], diff_variable=variable)

    def _unary_new_copy(self, child):
        """ See :meth:`UnaryOperator._unary_new_copy()`. """
        return FunctionParameter(self.name, child, diff_variable=self.diff_variable)
