#
# Parameter class
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

    def __init__(self, name, child):
        super().__init__(name, child)
