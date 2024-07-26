#
# Function classes and methods
#
from __future__ import annotations

import numpy as np
from scipy import special
import sympy
from typing import Callable
from collections.abc import Sequence
from typing_extensions import TypeVar

import pybamm


class Function(pybamm.Symbol):
    """
    A node in the expression tree representing an arbitrary function.

    Parameters
    ----------
    function : method
        A function can have 0 or many inputs. If no inputs are given, self.evaluate()
        simply returns func(). Otherwise, self.evaluate(t, y, u) returns
        func(child0.evaluate(t, y, u), child1.evaluate(t, y, u), etc).
    children : :class:`pybamm.Symbol`
        The children nodes to apply the function to
    differentiated_function : method, optional
        The function which was differentiated to obtain this one. Default is None.
    """

    def __init__(
        self,
        function: Callable,
        *children: pybamm.Symbol,
        name: str | None = None,
        differentiated_function: Callable | None = None,
    ):
        # Turn numbers into scalars
        children = list(children)
        for idx, child in enumerate(children):
            if isinstance(child, (float, int, np.number)):
                children[idx] = pybamm.Scalar(child)

        if name is not None:
            self.name = name
        else:
            try:
                name = f"function ({function.__name__})"
            except AttributeError:
                name = f"function ({function.__class__})"
        domains = self.get_children_domains(children)

        self.function = function
        self.differentiated_function = differentiated_function

        super().__init__(name, children=children, domains=domains)

    def __str__(self):
        """See :meth:`pybamm.Symbol.__str__()`."""
        out = f"{self.name[10:-1]}("
        for child in self.children:
            out += f"{child!s}, "
        out = out[:-2] + ")"
        return out

    def diff(self, variable: pybamm.Symbol):
        """See :meth:`pybamm.Symbol.diff()`."""
        if variable == self:
            return pybamm.Scalar(1)
        else:
            children = self.orphans
            partial_derivatives: list[None | pybamm.Symbol] = [None] * len(children)
            for i, child in enumerate(self.children):
                # if variable appears in the function, differentiate
                # function, and apply chain rule
                if variable in child.pre_order():
                    partial_derivatives[i] = self._function_diff(
                        children, i
                    ) * child.diff(variable)

            # remove None entries
            partial_derivatives = [x for x in partial_derivatives if x is not None]

            derivative = sum(partial_derivatives)
            if derivative == 0:
                return pybamm.Scalar(0)

            return derivative

    def _function_diff(self, children: Sequence[pybamm.Symbol], idx: float):
        """
        Derivative with respect to child number 'idx'.
        See :meth:`pybamm.Symbol._diff()`.
        """
        raise NotImplementedError(
            "Derivative of base Function class is not implemented. "
            "Please implement in child class."
        )

    def _function_jac(self, children_jacs):
        """Calculate the Jacobian of a function."""

        if all(child.evaluates_to_constant_number() for child in self.children):
            jacobian = pybamm.Scalar(0)
        else:
            # if at least one child contains variable dependence, then
            # calculate the required partial Jacobians and add them
            jacobian = None
            children = self.orphans
            for i, child in enumerate(children):
                if not child.evaluates_to_constant_number():
                    jac_fun = self._function_diff(children, i) * children_jacs[i]
                    jac_fun.clear_domains()
                    if jacobian is None:
                        jacobian = jac_fun
                    else:
                        jacobian += jac_fun

        return jacobian

    def evaluate(
        self,
        t: float | None = None,
        y: np.ndarray | None = None,
        y_dot: np.ndarray | None = None,
        inputs: dict | str | None = None,
    ):
        """See :meth:`pybamm.Symbol.evaluate()`."""
        evaluated_children = [
            child.evaluate(t, y, y_dot, inputs) for child in self.children
        ]
        return self._function_evaluate(evaluated_children)

    def _evaluates_on_edges(self, dimension: str) -> bool:
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return any(child.evaluates_on_edges(dimension) for child in self.children)

    def is_constant(self):
        """See :meth:`pybamm.Symbol.is_constant()`."""
        return all(child.is_constant() for child in self.children)

    def _evaluate_for_shape(self):
        """
        Default behaviour: has same shape as all child
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        evaluated_children = [child.evaluate_for_shape() for child in self.children]
        return self._function_evaluate(evaluated_children)

    def _function_evaluate(self, evaluated_children):
        return self.function(*evaluated_children)

    def create_copy(
        self,
        new_children: list[pybamm.Symbol] | None = None,
        perform_simplifications: bool = True,
    ):
        """See :meth:`pybamm.Symbol.new_copy()`."""
        children = self._children_for_copying(new_children)

        if not perform_simplifications:
            return pybamm.Function(
                self.function,
                *children,
                name=self.name,
                differentiated_function=self.differentiated_function,
            )
        else:
            # performs additional simplifications, rather than just calling the
            # constructor
            return self._function_new_copy(children)

    def _function_new_copy(self, children: list) -> Function:
        """
        Returns a new copy of the function.

        Inputs
        ------
        children : : list
            A list of the children of the function

        Returns
        -------
            : :pybamm.Function
            A new copy of the function
        """
        return pybamm.simplify_if_constant(
            pybamm.Function(
                self.function,
                *children,
                name=self.name,
                differentiated_function=self.differentiated_function,
            )
        )

    def _sympy_operator(self, child):
        """Apply appropriate SymPy operators."""
        return child

    def to_equation(self):
        """Convert the node and its subtree into a SymPy equation."""
        if self.print_name is not None:
            return sympy.Symbol(self.print_name)
        else:
            eq_list = []
            for child in self.children:
                eq = child.to_equation()
                eq_list.append(eq)
            return self._sympy_operator(*eq_list)

    def to_json(self):
        raise NotImplementedError(
            "pybamm.Function: Serialisation is only implemented for discretised models."
        )

    @classmethod
    def _from_json(cls, snippet):
        raise NotImplementedError(
            "pybamm.Function: Please use a discretised model when reading in from JSON."
        )


class SpecificFunction(Function):
    """
    Parent class for the specific functions, which implement their own `diff`
    operators directly.

    Parameters
    ----------
    function : method
        Function to be applied to child
    child : :class:`pybamm.Symbol`
        The child to apply the function to
    """

    def __init__(self, function: Callable, child: pybamm.Symbol):
        super().__init__(function, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """
        Reconstructs a SpecificFunction instance during deserialisation of a JSON file.

        Parameters
        ----------
        function : method
            Function to be applied to child
        snippet: dict
            Contains the child to apply the function to
        """

        instance = cls.__new__(cls)

        super(SpecificFunction, instance).__init__(
            snippet["function"], snippet["children"][0]
        )

        return instance

    def _function_new_copy(self, children):
        """See :meth:`pybamm.Function._function_new_copy()`"""
        return pybamm.simplify_if_constant(self.__class__(*children))

    def _sympy_operator(self, child):
        """Apply appropriate SymPy operators."""
        class_name = self.__class__.__name__.lower()
        sympy_function = getattr(sympy, class_name)
        return sympy_function(child)

    def to_json(self):
        """
        Method to serialise a SpecificFunction object into JSON.
        """

        json_dict = {
            "name": self.name,
            "id": self.id,
            "function": self.function.__name__,
        }

        return json_dict


SF = TypeVar("SF", bound=SpecificFunction)


def simplified_function(func_class: type[SF], child: pybamm.Symbol):
    """
    Simplifications implemented before applying the function.
    Currently only implemented for one-child functions.
    """
    if isinstance(child, pybamm.Broadcast):
        # Move the function inside the broadcast
        # Apply recursively
        func_child_not_broad = pybamm.simplify_if_constant(
            simplified_function(func_class, child.orphans[0])
        )
        return child._unary_new_copy(func_child_not_broad)
    else:
        return pybamm.simplify_if_constant(func_class(child))  # type: ignore[call-arg, arg-type]


class Arcsinh(SpecificFunction):
    """Arcsinh function."""

    def __init__(self, child):
        super().__init__(np.arcsinh, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = np.arcsinh
        instance = super()._from_json(snippet)
        return instance

    def _function_diff(self, children, idx):
        """See :meth:`pybamm.Symbol._function_diff()`."""
        return 1 / sqrt(children[0] ** 2 + 1)

    def _sympy_operator(self, child):
        """Override :meth:`pybamm.Function._sympy_operator`"""
        return sympy.asinh(child)


def arcsinh(child: pybamm.Symbol):
    """Returns arcsinh function of child."""
    return simplified_function(Arcsinh, child)


class Arctan(SpecificFunction):
    """Arctan function."""

    def __init__(self, child):
        super().__init__(np.arctan, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = np.arctan
        instance = super()._from_json(snippet)
        return instance

    def _function_diff(self, children, idx):
        """See :meth:`pybamm.Function._function_diff()`."""
        return 1 / (children[0] ** 2 + 1)

    def _sympy_operator(self, child):
        """Override :meth:`pybamm.Function._sympy_operator`"""
        return sympy.atan(child)


def arctan(child: pybamm.Symbol):
    """Returns hyperbolic tan function of child."""
    return simplified_function(Arctan, child)


class Cos(SpecificFunction):
    """Cosine function."""

    def __init__(self, child):
        super().__init__(np.cos, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = np.cos
        instance = super()._from_json(snippet)
        return instance

    def _function_diff(self, children, idx):
        """See :meth:`pybamm.Symbol._function_diff()`."""
        return -sin(children[0])


def cos(child: pybamm.Symbol):
    """Returns cosine function of child."""
    return simplified_function(Cos, child)


class Cosh(SpecificFunction):
    """Hyberbolic cosine function."""

    def __init__(self, child):
        super().__init__(np.cosh, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = np.cosh
        instance = super()._from_json(snippet)
        return instance

    def _function_diff(self, children, idx):
        """See :meth:`pybamm.Function._function_diff()`."""
        return sinh(children[0])


def cosh(child: pybamm.Symbol):
    """Returns hyperbolic cosine function of child."""
    return simplified_function(Cosh, child)


class Erf(SpecificFunction):
    """Error function."""

    def __init__(self, child):
        super().__init__(special.erf, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = special.erf
        instance = super()._from_json(snippet)
        return instance

    def _function_diff(self, children, idx):
        """See :meth:`pybamm.Function._function_diff()`."""
        return 2 / np.sqrt(np.pi) * exp(-(children[0] ** 2))


def erf(child: pybamm.Symbol):
    """Returns error function of child."""
    return simplified_function(Erf, child)


def erfc(child: pybamm.Symbol):
    """Returns complementary error function of child."""
    return 1 - simplified_function(Erf, child)


class Exp(SpecificFunction):
    """Exponential function."""

    def __init__(self, child):
        super().__init__(np.exp, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = np.exp
        instance = super()._from_json(snippet)
        return instance

    def _function_diff(self, children, idx):
        """See :meth:`pybamm.Function._function_diff()`."""
        return exp(children[0])


def exp(child: pybamm.Symbol):
    """Returns exponential function of child."""
    return simplified_function(Exp, child)


class Log(SpecificFunction):
    """Logarithmic function."""

    def __init__(self, child):
        super().__init__(np.log, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = np.log
        instance = super()._from_json(snippet)
        return instance

    def _function_evaluate(self, evaluated_children):
        # don't raise RuntimeWarning for NaNs
        with np.errstate(invalid="ignore"):
            return np.log(*evaluated_children)

    def _function_diff(self, children, idx):
        """See :meth:`pybamm.Function._function_diff()`."""
        return 1 / children[0]


def log(child, base="e"):
    """Returns logarithmic function of child (any base, default 'e')."""
    log_child = simplified_function(Log, child)
    if base == "e":
        return log_child
    else:
        return log_child / np.log(base)


def log10(child: pybamm.Symbol):
    """Returns logarithmic function of child, with base 10."""
    return log(child, base=10)


class Max(SpecificFunction):
    """Max function."""

    def __init__(self, child):
        super().__init__(np.max, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = np.max
        instance = super()._from_json(snippet)
        return instance

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        # Max will always return a scalar
        return np.nan * np.ones((1, 1))


def max(child: pybamm.Symbol):
    """
    Returns max function of child. Not to be confused with :meth:`pybamm.maximum`, which
    returns the larger of two objects.
    """
    return pybamm.simplify_if_constant(Max(child))


class Min(SpecificFunction):
    """Min function."""

    def __init__(self, child):
        super().__init__(np.min, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = np.min
        instance = super()._from_json(snippet)
        return instance

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        # Min will always return a scalar
        return np.nan * np.ones((1, 1))


def min(child: pybamm.Symbol):
    """
    Returns min function of child. Not to be confused with :meth:`pybamm.minimum`, which
    returns the smaller of two objects.
    """
    return pybamm.simplify_if_constant(Min(child))


def sech(child: pybamm.Symbol):
    """Returns hyperbolic sec function of child."""
    return 1 / simplified_function(Cosh, child)


class Sin(SpecificFunction):
    """Sine function."""

    def __init__(self, child):
        super().__init__(np.sin, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = np.sin
        instance = super()._from_json(snippet)
        return instance

    def _function_diff(self, children, idx):
        """See :meth:`pybamm.Function._function_diff()`."""
        return cos(children[0])


def sin(child: pybamm.Symbol):
    """Returns sine function of child."""
    return simplified_function(Sin, child)


class Sinh(SpecificFunction):
    """Hyperbolic sine function."""

    def __init__(self, child):
        super().__init__(np.sinh, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = np.sinh
        instance = super()._from_json(snippet)
        return instance

    def _function_diff(self, children, idx):
        """See :meth:`pybamm.Function._function_diff()`."""
        return cosh(children[0])


def sinh(child: pybamm.Symbol):
    """Returns hyperbolic sine function of child."""
    return simplified_function(Sinh, child)


class Sqrt(SpecificFunction):
    """Square root function."""

    def __init__(self, child):
        super().__init__(np.sqrt, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = np.sqrt
        instance = super()._from_json(snippet)
        return instance

    def _function_evaluate(self, evaluated_children):
        # don't raise RuntimeWarning for NaNs
        with np.errstate(invalid="ignore"):
            return np.sqrt(*evaluated_children)

    def _function_diff(self, children, idx):
        """See :meth:`pybamm.Function._function_diff()`."""
        return 1 / (2 * sqrt(children[0]))


def sqrt(child: pybamm.Symbol):
    """Returns square root function of child."""
    return simplified_function(Sqrt, child)


class Tanh(SpecificFunction):
    """Hyperbolic tan function."""

    def __init__(self, child):
        super().__init__(np.tanh, child)

    @classmethod
    def _from_json(cls, snippet: dict):
        """See :meth:`pybamm.SpecificFunction._from_json()`."""
        snippet["function"] = np.tanh
        instance = super()._from_json(snippet)
        return instance

    def _function_diff(self, children, idx):
        """See :meth:`pybamm.Function._function_diff()`."""
        return sech(children[0]) ** 2


def tanh(child: pybamm.Symbol):
    """Returns hyperbolic tan function of child."""
    return simplified_function(Tanh, child)


def normal_pdf(
    x: pybamm.Symbol, mu: pybamm.Symbol | float, sigma: pybamm.Symbol | float
):
    """
    Returns the normal probability density function at x.

    Parameters
    ----------
    x : pybamm.Symbol
        The value at which to evaluate the normal distribution
    mu : pybamm.Symbol or float
        The mean of the normal distribution
    sigma : pybamm.Symbol or float
        The standard deviation of the normal distribution

    Returns
    -------
    pybamm.Symbol
        The value of the normal distribution at x
    """
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def normal_cdf(
    x: pybamm.Symbol, mu: pybamm.Symbol | float, sigma: pybamm.Symbol | float
):
    """
    Returns the normal cumulative distribution function at x.

    Parameters
    ----------
    x : pybamm.Symbol
        The value at which to evaluate the normal distribution
    mu : pybamm.Symbol or float
        The mean of the normal distribution
    sigma : pybamm.Symbol or float
        The standard deviation of the normal distribution

    Returns
    -------
    pybamm.Symbol
        The value of the normal distribution at x
    """
    return 0.5 * (1 + special.erf((x - mu) / (sigma * np.sqrt(2))))
