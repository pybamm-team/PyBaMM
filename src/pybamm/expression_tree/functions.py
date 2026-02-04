#
# Function classes and methods
#
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import sympy
from scipy import special
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
            if isinstance(child, float | int | np.number):
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
        y: npt.NDArray[np.float64] | None = None,
        y_dot: npt.NDArray[np.float64] | None = None,
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


class Arcsinh2(Function):
    """
    Two-argument arcsinh function for arcsinh(a/b) that avoids division by zero
    by adding a small regularisation term to the denominator.

    Computes arcsinh(a / b_eff) where b_eff = sign(b) * hypot(b, eps).
    Note: the sign(b) function treats sign(0) as 1 for numerical stability.

    Parameters
    ----------
    a : pybamm.Symbol or float
        The numerator argument
    b : pybamm.Symbol or float
        The denominator argument
    eps : float, optional
        Small regularisation parameter. Defaults to
        pybamm.settings.tolerances["reg_arcsinh2"]

    Returns
    -------
    pybamm.Symbol
        The regularised arcsinh(a/b) value
    """

    def __init__(
        self,
        a: pybamm.Symbol,
        b: pybamm.Symbol,
        eps: float | None = None,
    ):
        if eps is None:
            eps = pybamm.settings.tolerances["reg_arcsinh2"]
        self.eps = eps
        super().__init__(self._arcsinh2_evaluate, a, b, name="arcsinh2")

    @staticmethod
    def _arcsinh2_evaluate(a, b, eps):
        """Evaluate arcsinh2 using numpy.

        Computes arcsinh(a/b) with regularization to avoid division by zero.
        Uses arcsinh(a / b_eff) where b_eff = sign(b) * hypot(b, eps).
        This formula has the correct derivative 1/b_eff at a=0.
        """
        # sign(b) but treat sign(0) as non-zero
        sign_b = np.where(b >= 0, 1.0, -1.0)
        b_eff = sign_b * np.hypot(b, eps)
        return np.arcsinh(a / b_eff)

    def _function_evaluate(self, evaluated_children):
        """See :meth:`pybamm.Function._function_evaluate()`."""
        return self._arcsinh2_evaluate(
            evaluated_children[0], evaluated_children[1], self.eps
        )

    def _function_diff(self, children, idx):
        """
        Derivative with respect to child number 'idx'.

        For f(a, b) = arcsinh(a / b_eff) where b_eff = sign(b) * hypot(b, eps):

        df/da = sign(b) / hypot(a, hypot(b, eps))
        df/db = -a * |b| / (hypot(a, hypot(b, eps)) * hypot(b, eps)^2)

        Note: sign(b) = 1 when b >= 0.
        """
        a, b = children
        b_eff = pybamm.hypot(b, self.eps)  # |b_eff| = hypot(b, eps), always positive
        h = pybamm.hypot(a, b_eff)

        # sign(b) but treat sign(0) as non-zero
        sign_b = 2 * (b >= 0) - 1

        if idx == 0:
            # df/da = sign(b) / hypot(a, |b_eff|)
            return sign_b / h
        elif idx == 1:
            # df/db = -a * |b| / (h * |b_eff|^2)
            return -a * abs(b) / (h * b_eff**2)
        else:
            raise IndexError("Arcsinh2 only has two children (a, b)")

    def _function_new_copy(self, children):
        """See :meth:`pybamm.Function._function_new_copy()`"""
        return Arcsinh2(*children, eps=self.eps)

    def _sympy_operator(self, a, b):
        """Convert to SymPy expression."""
        # sign(b) but treat sign(0) as non-zero
        sign_b = sympy.Piecewise((1, b >= 0), (-1, True))
        b_eff = sign_b * sympy.sqrt(b**2 + self.eps**2)
        return sympy.asinh(a / b_eff)

    def to_json(self):
        """Method to serialise a Arcsinh2 object into JSON."""
        return {
            "name": self.name,
            "id": self.id,
            "function": "arcsinh2",
            "eps": self.eps,
        }

    @classmethod
    def _from_json(cls, snippet: dict):
        """Reconstruct a Arcsinh2 instance from JSON."""
        instance = cls.__new__(cls)
        instance.eps = snippet.get("eps", pybamm.settings.tolerances["reg_arcsinh2"])
        # The parent Function.__init__ will be called with children from snippet
        super(Arcsinh2, instance).__init__(
            instance._arcsinh2_evaluate,
            *snippet["children"],
            name="arcsinh2",
        )
        return instance


def arcsinh2(
    a: pybamm.Symbol | float,
    b: pybamm.Symbol | float,
    eps: float | None = None,
) -> pybamm.Symbol:
    """
    Two-argument arcsinh function for arcsinh(a/b) that avoids division by zero
    by adding a small regularisation term to the denominator.

    Computes arcsinh(a / b_eff), where b_eff = sign(b) * hypot(b, eps).

    Parameters
    ----------
    a : pybamm.Symbol or float
        The numerator argument
    b : pybamm.Symbol or float
        The denominator argument
    eps : float, optional
        Small regularisation parameter. Defaults to
        pybamm.settings.tolerances["reg_arcsinh2"]

    Returns
    -------
    pybamm.Symbol
        The regularised arcsinh(a/b) value
    """
    # Convert scalars to pybamm types
    a = pybamm.convert_to_symbol(a)
    b = pybamm.convert_to_symbol(b)
    return Arcsinh2(a, b, eps=eps)


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


class RegPower(Function):
    """
    Regularised power: |x|^a * sign(x) with finite derivative at x=0.

    Approximates |x|^a * sign(x) using:
        y = x * (x^2 + delta^2)^((a-1)/2)

    When scale is set:
        y = (x/scale) * ((x/scale)^2 + delta^2)^((a-1)/2) * scale^a

    Behavior:
    - For |x| >> delta: returns |x|^a * sign(x)
    - For |x| << delta: returns x * delta^(a-1) (linear)
    - Smooth transition in between

    This is an anti-symmetric function: RegPower(-x, a) = -RegPower(x, a)

    Parameters
    ----------
    base : :class:`pybamm.Symbol`
        Base expression (x)
    exponent : :class:`pybamm.Symbol` or float
        Exponent expression (a)
    scale : :class:`pybamm.Symbol` or float, optional
        Scale factor for the input. Defaults to 1 (no scaling).

    References
    ----------
    .. [1] Modelica.Fluid.Utilities.regPow
    """

    def __init__(
        self,
        base: pybamm.Symbol | float,
        exponent: pybamm.Symbol | float,
        scale: pybamm.Symbol | float | None = None,
        delta: float | None = None,
    ):
        # Convert to symbols
        base = pybamm.convert_to_symbol(base)
        exponent = pybamm.convert_to_symbol(exponent)
        if scale is None:
            scale = pybamm.Scalar(1)
        else:
            scale = pybamm.convert_to_symbol(scale)

        if delta is None:
            delta = pybamm.settings.tolerances["reg_power"]
        self.delta = delta

        super().__init__(
            self._reg_power_evaluate, base, exponent, scale, name="reg_power"
        )

    def _reg_power_evaluate(self, base, exponent, scale):
        """Evaluate reg_power using numpy."""
        x = base / scale
        x2_d2 = x**2 + self.delta**2
        return x * (x2_d2 ** ((exponent - 1) / 2)) * (scale**exponent)

    def _function_evaluate(self, evaluated_children):
        """See :meth:`pybamm.Function._function_evaluate()`."""
        return self._reg_power_evaluate(*evaluated_children)

    def _function_diff(self, children, idx):
        """
        Derivative with respect to child number 'idx'.

        Children are: [base, exponent, scale]
        """
        base, exponent, scale = children
        delta = self.delta

        x = base / scale
        x2_d2 = x**2 + delta**2
        scale_factor = scale**exponent

        if idx == 0:
            # Derivative w.r.t. base
            # d/dx [x * (x^2 + d^2)^((a-1)/2)] = (x^2 + d^2)^((a-3)/2) * (a*x^2 + d^2)
            dreg_dx = (x2_d2 ** ((exponent - 3) / 2)) * (exponent * x**2 + delta**2)
            # Chain rule: d/d(base) = dreg_dx * (1/scale) * scale^a = dreg_dx * scale^(a-1)
            return dreg_dx * (scale ** (exponent - 1))
        elif idx == 1:
            # Derivative w.r.t. exponent
            reg_val = x * (x2_d2 ** ((exponent - 1) / 2)) * scale_factor
            # d/da = reg_val * (log(x^2 + d^2) / 2 + log(scale))
            return reg_val * (pybamm.log(x2_d2) / 2 + pybamm.log(scale))
        elif idx == 2:
            # Derivative w.r.t. scale
            # y = x * (x^2 + d^2)^((a-1)/2) * scale^a, where x = base/scale
            # Using quotient rule and chain rule:
            # dy/dscale = -base/scale^2 * (x^2+d^2)^((a-3)/2) * (a*x^2 + d^2) * scale^a
            #           + x * (x^2+d^2)^((a-1)/2) * a * scale^(a-1)
            # Simplified:
            reg_val = x * (x2_d2 ** ((exponent - 1) / 2)) * scale_factor
            dreg_dx = (x2_d2 ** ((exponent - 3) / 2)) * (exponent * x**2 + delta**2)
            # dy/dscale = reg_val * a / scale - dreg_dx * x * scale^(a-1)
            #           = reg_val * a / scale - dreg_dx * base / scale * scale^(a-1) / scale
            #           = (reg_val * a - base * dreg_dx * scale^(a-1)) / scale
            return (
                reg_val * exponent - base * dreg_dx * (scale ** (exponent - 1))
            ) / scale
        else:
            raise IndexError("RegPower has three children (base, exponent, scale)")

    def _function_jac(self, children_jacs):
        """See :meth:`pybamm.Function._function_jac()`."""
        children = self.orphans
        base, exponent, scale = children
        delta = self.delta

        x = base / scale
        x2_d2 = x**2 + delta**2
        scale_factor = scale**exponent

        jacobian = None

        # Base Jacobian
        if not base.evaluates_to_constant_number():
            dreg_dx = (x2_d2 ** ((exponent - 3) / 2)) * (exponent * x**2 + delta**2)
            dreg_dbase = dreg_dx * (scale ** (exponent - 1))
            jac_term = dreg_dbase * children_jacs[0]
            jac_term.clear_domains()
            jacobian = jac_term

        # Exponent Jacobian
        if not exponent.evaluates_to_constant_number():
            reg_val = x * (x2_d2 ** ((exponent - 1) / 2)) * scale_factor
            dreg_da = reg_val * (pybamm.log(x2_d2) / 2 + pybamm.log(scale))
            jac_term = dreg_da * children_jacs[1]
            jac_term.clear_domains()
            if jacobian is None:
                jacobian = jac_term
            else:
                jacobian += jac_term

        # Scale Jacobian
        if not scale.evaluates_to_constant_number():
            reg_val = x * (x2_d2 ** ((exponent - 1) / 2)) * scale_factor
            dreg_dx = (x2_d2 ** ((exponent - 3) / 2)) * (exponent * x**2 + delta**2)
            dreg_dscale = (
                reg_val * exponent - base * dreg_dx * (scale ** (exponent - 1))
            ) / scale
            jac_term = dreg_dscale * children_jacs[2]
            jac_term.clear_domains()
            if jacobian is None:
                jacobian = jac_term
            else:
                jacobian += jac_term

        if jacobian is None:
            jacobian = pybamm.Scalar(0)

        return jacobian

    def _function_new_copy(self, children):
        """See :meth:`pybamm.Function._function_new_copy()`"""
        base, exponent, scale = children
        return pybamm.simplify_if_constant(
            RegPower(base, exponent, scale=scale, delta=self.delta)
        )

    def _sympy_operator(self, base, exponent, scale):
        """Convert to SymPy expression."""
        x = base / scale
        x2_d2 = x**2 + self.delta**2
        return x * sympy.Pow(x2_d2, (exponent - 1) / 2) * sympy.Pow(scale, exponent)

    def __str__(self):
        """See :meth:`pybamm.Symbol.__str__()`."""
        base, exponent, scale = self.children
        return f"reg_power({base!s}, {exponent!s}, scale={scale!s})"

    def to_json(self):
        """Method to serialise a RegPower object into JSON."""
        return {
            "name": self.name,
            "id": self.id,
            "function": "reg_power",
            "delta": self.delta,
        }

    @classmethod
    def _from_json(cls, snippet: dict):
        """Reconstruct a RegPower instance from JSON."""
        instance = cls.__new__(cls)
        instance.delta = snippet.get("delta", pybamm.settings.tolerances["reg_power"])
        super(RegPower, instance).__init__(
            instance._reg_power_evaluate,
            *snippet["children"],
            name="reg_power",
        )
        return instance


def reg_power(
    base: pybamm.Symbol | float,
    exponent: pybamm.Symbol | float,
    scale: pybamm.Symbol | float | None = None,
) -> pybamm.Symbol:
    """
    Regularised power: |x|^a * sign(x) with finite derivative at x=0.

    Convenience function that creates a :class:`RegPower` node.

    Parameters
    ----------
    base : :class:`pybamm.Symbol` or float
        Input expression (x)
    exponent : float
        Power exponent (must be > 0)
    scale : float, optional
        Scale factor for the input. Defaults to 1 (no scaling).

    Returns
    -------
    :class:`RegPower`
        Regularised power node

    References
    ----------
    .. [1] Modelica.Fluid.Utilities.regPow
    """
    return pybamm.simplify_if_constant(RegPower(base, exponent, scale=scale))


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
