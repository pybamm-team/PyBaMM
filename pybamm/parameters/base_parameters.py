#
# Base parameters class
#
import pybamm


class BaseParameters:
    """
    Overload the `__setattr__` method to record what the variable was called.
    """

    def __setattr__(self, name, value):
        if isinstance(value, pybamm.Symbol):
            value.print_name = name
        super().__setattr__(name, value)
