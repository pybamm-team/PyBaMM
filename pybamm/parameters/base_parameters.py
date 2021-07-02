#
# Base parameters class
#
import pybamm
from pybamm.expression_tree.print_name import prettify_print_name


class BaseParameters:
    """
    Overload the `__setattr__` method to record what the variable was called.
    """

    def __setattr__(self, name, value):
        if isinstance(value, pybamm.Symbol):
            value.print_name = prettify_print_name(name)
        super().__setattr__(name, value)
