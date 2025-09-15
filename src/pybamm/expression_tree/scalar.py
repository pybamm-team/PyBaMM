#
# Scalar class
#
from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import sympy

import pybamm
from pybamm.type_definitions import Numeric


class Scalar(pybamm.Symbol):
    """
    A node in the expression tree representing a scalar value.

    Parameters
    ----------
    value : numeric
        the value returned by the node when evaluated
    name : str, optional
        the name of the node. Defaulted to ``str(value)`` if not provided


    """

    def __init__(
        self,
        value: Numeric,
        name: str | None = None,
    ) -> None:
        # set default name if not provided
        self.value = value
        if name is None:
            name = str(self.value)

        super().__init__(name)

    @classmethod
    def _from_json(cls, snippet: dict):
        return cls(snippet["value"], name=snippet["name"])

    def __str__(self):
        return str(self.value)

    @property
    def value(self):
        """The value returned by the node when evaluated."""
        return self._value

    # address numpy 1.25 deprecation warning: array should have ndim=0 before conversion
    @value.setter
    def value(self, value):
        self._value = (
            np.float64(value.item())
            if isinstance(value, np.ndarray)
            else np.float64(value)
        )

    def set_id(self):
        """See :meth:`pybamm.Symbol.set_id()`."""
        # We must include the value in the hash, since different scalars can be
        # indistinguishable by class and name alone
        self._id = hash((self.__class__, str(self.value)))

    def _base_evaluate(
        self,
        t: float | None = None,
        y: npt.NDArray[np.float64] | None = None,
        y_dot: npt.NDArray[np.float64] | None = None,
        inputs: dict | str | None = None,
    ):
        """See :meth:`pybamm.Symbol._base_evaluate()`."""
        return self._value

    def _jac(self, variable: pybamm.Variable) -> pybamm.Scalar:
        """See :meth:`pybamm.Symbol._jac()`."""
        return pybamm.Scalar(0)

    def create_copy(
        self,
        new_children=None,
        perform_simplifications=True,
    ):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        if new_children is None or len(new_children) == 0:
            return Scalar(self.value, self.name)
        else:
            raise ValueError(
                f"Cannot create a copy of a scalar with new children: {new_children}"
            )

    def is_constant(self) -> Literal[True]:
        """See :meth:`pybamm.Symbol.is_constant()`."""
        return True

    def to_equation(self):
        """Returns the value returned by the node when evaluated."""
        if self.print_name is not None:
            return sympy.Symbol(self.print_name)
        else:
            return self.value

    def to_json(self):
        """
        Method to serialise a Symbol object into JSON.
        """

        json_dict = {"name": self.name, "id": self.id, "value": self.value}

        return json_dict


class Constant(Scalar):
    """
    A node in the expression tree representing a named constant value. This is a subclass of
    :class:`pybamm.Scalar` and is used to represent named constants such as :math:`R`, :math:`F`,
    and :math:`k_b`. Counterintuitively, the "is_constant" method is set to False, so that
    named constants are not simplified when constructed.

    Parameters
    ----------
    value : numeric
        the value returned by the node when evaluated
    name : str
        the name of the node
    """

    def __init__(self, value: Numeric, name: str):
        super().__init__(value, name)

    def is_constant(self) -> Literal[True]:
        """
        The value is constant, but setting this to False makes sure we retain named constants
        in the expression tree. For example, `pybamm.constants.R + 2` should not be
        simplified to `10.314462618` until after the expression is evaluated. See :meth:`pybamm.Symbol.is_constant()`.
        """
        return False

    def __str__(self):
        return str(self.name)
