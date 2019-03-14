#
# Function Parameter class
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class FunctionParameter(pybamm.UnaryOperator):
    """A node in the expression tree representing a function parameter

    This node will be replaced by a :class:`.Function` node

    Parameters
    ----------

    name : str
        name of the node
    child : :class:`Symbol`
        child node
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list

    """

    def __init__(self, name, child, diff_variable=None):
        super().__init__(name, child)
        # assign diff variable
        self.diff_variable = diff_variable

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        # return a new FunctionParameter, that knows it will need to be differentiated
        # when the parameters are set
        return FunctionParameter(self.name, self.orphans[0], diff_variable=variable)
