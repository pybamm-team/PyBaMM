#
# Scalar class
#
import numpy as np
import sympy

import pybamm


class Scalar(pybamm.Symbol):
    """
    A node in the expression tree representing a scalar value.

    Parameters
    ----------
    value : numeric or str
        the value returned by the node when evaluated
    units : str, optional
        The units of the symbol. If not provided, the units are assumed to be
        dimensionless.
    name : str, optional
        the name of the node. Defaulted to ``str(value) [units]`` if not provided

    **Extends:** :class:`Symbol`
    """

    def __init__(self, value, units=None, name=None):
        # value can be given as a string with units (e.g. "1.5 [A]")
        if isinstance(value, str) and "[" in value and "]" in value:
            if units is not None:
                raise pybamm.UnitsError(
                    "Cannot provide units as both a string and a separate argument"
                )
            value, units = value.split(" [")
            value = float(value)
            units = units[:-1]

        self.value = value
        # set default name if not provided
        if name is None:
            name = str(self.value)
            if not (
                units is None
                or (isinstance(units, pybamm.Units) and units.units_dict == {})
            ):
                name += f" [{str(units)}]"

        super().__init__(name)

    def __str__(self):
        return str(self.value)

    @property
    def value(self):
        """The value returned by the node when evaluated."""
        return self._value

    @value.setter
    def value(self, value):
        self._value = np.float64(value)

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`."""
        # We must include the value in the hash, since different scalars can be
        # indistinguishable by class and name alone
        self._id = hash((self.__class__, self.name) + tuple(str(self._value)))

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """See :meth:`pybamm.Symbol._base_evaluate()`."""
        return self._value

    def _jac(self, variable):
        """See :meth:`pybamm.Symbol._jac()`."""
        return pybamm.Scalar(0)

    def create_copy(self):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        return Scalar(self.value, units=self.units, name=self.name)

    def is_constant(self):
        """See :meth:`pybamm.Symbol.is_constant()`."""
        return True

    def to_equation(self):
        """Returns the value returned by the node when evaluated."""
        if self.print_name is not None:
            return sympy.Symbol(self.print_name)
        else:
            return self.value
