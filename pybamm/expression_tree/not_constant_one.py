#
# Special class for a symbol that should not be treated as a constant, and evaluates
# to 1
#
import numbers
import numpy as np
import pybamm


class NotConstantOne(pybamm.Symbol):
    """Special class for a symbol that should not be treated as a constant, and
    evaluates to 1.
    """

    def __init__(self):
        super().__init__("not_constant_one")

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return NotConstantOne()

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """ See :meth:`pybamm.Symbol._base_evaluate()`. """
        # Default value of 1
        return 1

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
        return pybamm.Scalar(0)

    def is_constant(self):
        """ See :meth:`pybamm.Symbol.is_constant()`. """
        # This symbol is not constant
        return False
