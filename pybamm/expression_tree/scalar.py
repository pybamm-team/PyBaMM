#
# Scalar class
#
import pybamm
import numpy as np


class Scalar(pybamm.Symbol):
    """A node in the expression tree representing a scalar value

    **Extends:** :class:`Symbol`

    Parameters
    ----------

    value : numeric
        the value returned by the node when evaluated
    name : str, optional
        the name of the node. Defaulted to ``str(value)``
        if not provided
    domain : iterable of str, optional
        list of domains the parameter is valid over, defaults to empty list

    """

    def __init__(self, value, name=None, domain=[]):
        # set default name if not provided
        self.value = value
        if name is None:
            name = str(self.value)

        super().__init__(name, domain=domain)

    @property
    def value(self):
        """the value returned by the node when evaluated"""
        return self._value

    @value.setter
    def value(self, value):
        self._value = np.float64(value)

    def set_id(self):
        """ See :meth:`pybamm.Symbol.set_id()`. """
        # We must include the value in the hash, since different scalars can be
        # indistinguishable by class, name and domain alone
        self._id = hash(
            (self.__class__, self.name) + tuple(self.domain) + tuple(str(self._value))
        )

    def _base_evaluate(self, t=None, y=None, y_dot=None, inputs=None):
        """ See :meth:`pybamm.Symbol._base_evaluate()`. """
        return self._value

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """
        return pybamm.Scalar(0)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        return Scalar(self.value, self.name, self.domain)

    def is_constant(self):
        """ See :meth:`pybamm.Symbol.is_constant()`. """
        return True
