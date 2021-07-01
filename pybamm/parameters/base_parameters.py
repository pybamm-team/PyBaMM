#
# Base parameters class
#
import pybamm
from pybamm.expression_tree.print_name_overrides import PRINT_NAME_OVERRIDES


class BaseParameters:
    """
    Overload the `__setattr__` method to record what the variable was called.
    """

    def __setattr__(self, name, value):
        if isinstance(value, pybamm.Symbol):
            value.print_name = PRINT_NAME_OVERRIDES.get(name, name)
        super().__setattr__(name, value)
