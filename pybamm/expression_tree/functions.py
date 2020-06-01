#
# Function classes and methods
#
import autograd
import numbers
import numpy as np
import pybamm


class Function(pybamm.Symbol):
    """A node in the expression tree representing an arbitrary function

    Parameters
    ----------
    function : method
        A function can have 0 or many inputs. If no inputs are given, self.evaluate()
        simply returns func(). Otherwise, self.evaluate(t, y, u) returns
        func(child0.evaluate(t, y, u), child1.evaluate(t, y, u), etc).
    children : :class:`pybamm.Symbol`
        The children nodes to apply the function to
    derivative : str, optional
        Which derivative to use when differentiating ("autograd" or "derivative").
        Default is "autograd".
    differentiated_function : method, optional
        The function which was differentiated to obtain this one. Default is None.
    **Extends:** :class:`pybamm.Symbol`
    """

    def __init__(
        self,
        function,
        *children,
        name=None,
        derivative="autograd",
        differentiated_function=None
    ):
        # Turn numbers into scalars
        children = list(children)
        for idx, child in enumerate(children):
            if isinstance(child, numbers.Number):
                children[idx] = pybamm.Scalar(child)

        if name is not None:
            self.name = name
        else:
            try:
                name = "function ({})".format(function.__name__)
            except AttributeError:
                name = "function ({})".format(function.__class__)
        domain = self.get_children_domains(children)
        auxiliary_domains = self.get_children_auxiliary_domains(children)

        self.function = function
        self.derivative = derivative
        self.differentiated_function = differentiated_function

        super().__init__(
            name, children=children, domain=domain, auxiliary_domains=auxiliary_domains
        )

    def get_children_domains(self, children_list):
        """Obtains the unique domain of the children. If the
        children have different domains then raise an error"""

        domains = [child.domain for child in children_list if child.domain != []]

        # check that there is one common domain amongst children
        distinct_domains = set(tuple(dom) for dom in domains)

        if len(distinct_domains) > 1:
            raise pybamm.DomainError(
                "Functions can only be applied to variables on the same domain"
            )
        elif len(distinct_domains) == 0:
            domain = []
        else:
            domain = domains[0]

        return domain

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            children = self.orphans
            partial_derivatives = [None] * len(children)
            for i, child in enumerate(self.children):
                # if variable appears in the function, differentiate
                # function, and apply chain rule
                if variable.id in [symbol.id for symbol in child.pre_order()]:
                    partial_derivatives[i] = self._function_diff(
                        children, i
                    ) * child.diff(variable)

            # remove None entries
            partial_derivatives = list(filter(None, partial_derivatives))

            derivative = sum(partial_derivatives)
            if derivative == 0:
                derivative = pybamm.Scalar(0)

            return derivative

    def _function_diff(self, children, idx):
        """
        Derivative with respect to child number 'idx'.
        See :meth:`pybamm.Symbol._diff()`.
        """
        # Store differentiated function, needed in case we want to convert to CasADi
        if self.derivative == "autograd":
            return Function(
                autograd.elementwise_grad(self.function, idx),
                *children,
                differentiated_function=self.function
            )
        elif self.derivative == "derivative":
            if len(children) > 1:
                raise ValueError(
                    """
                    differentiation using '.derivative()' not implemented for functions
                    with more than one child
                    """
                )
            else:
                # keep using "derivative" as derivative
                return pybamm.Function(
                    self.function.derivative(),
                    *children,
                    derivative="derivative",
                    differentiated_function=self.function
                )

    def _function_jac(self, children_jacs):
        """ Calculate the jacobian of a function. """

        if all(child.evaluates_to_number() for child in self.children):
            jacobian = pybamm.Scalar(0)
        else:
            # if at least one child contains variable dependence, then
            # calculate the required partial jacobians and add them
            jacobian = None
            children = self.orphans
            for i, child in enumerate(children):
                if not child.evaluates_to_number():
                    jac_fun = self._function_diff(children, i) * children_jacs[i]
                    jac_fun.clear_domains()
                    if jacobian is None:
                        jacobian = jac_fun
                    else:
                        jacobian += jac_fun

        return jacobian

    def evaluate(self, t=None, y=None, y_dot=None, inputs=None, known_evals=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        if known_evals is not None:
            if self.id not in known_evals:
                evaluated_children = [None] * len(self.children)
                for i, child in enumerate(self.children):
                    evaluated_children[i], known_evals = child.evaluate(
                        t, y, y_dot, inputs, known_evals=known_evals
                    )
                known_evals[self.id] = self._function_evaluate(evaluated_children)
            return known_evals[self.id], known_evals
        else:
            evaluated_children = [
                child.evaluate(t, y, y_dot, inputs) for child in self.children
            ]
            return self._function_evaluate(evaluated_children)

    def evaluates_on_edges(self):
        """ See :meth:`pybamm.Symbol.evaluates_on_edges()`. """
        return any(child.evaluates_on_edges() for child in self.children)

    def _evaluate_for_shape(self):
        """
        Default behaviour: has same shape as all child
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        evaluated_children = [child.evaluate_for_shape() for child in self.children]
        return self._function_evaluate(evaluated_children)

    def _function_evaluate(self, evaluated_children):
        return self.function(*evaluated_children)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        children_copy = [child.new_copy() for child in self.children]
        return self._function_new_copy(children_copy)

    def _function_new_copy(self, children):
        """Returns a new copy of the function.

        Inputs
        ------
        children : : list
            A list of the children of the function

        Returns
        -------
            : :pybamm.Function
            A new copy of the function
        """
        return pybamm.Function(
            self.function,
            *children,
            name=self.name,
            derivative=self.derivative,
            differentiated_function=self.differentiated_function
        )

    def _function_simplify(self, simplified_children):
        """
        Simplifies the function.

        Inputs
        ------
        simplified_children: : list
            A list of simplified children of the function

        Returns
        -------
         :: pybamm.Scalar() if no children
         :: pybamm.Function if there are children
        """
        return self._function_new_copy(simplified_children)


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

    def __init__(self, function, child):
        super().__init__(function, child)

    def _function_new_copy(self, children):
        """ See :meth:`pybamm.Function._function_new_copy()` """
        return self.__class__(*children)

    def _function_simplify(self, simplified_children):
        """ See :meth:`pybamm.Function._function_simplify()` """
        return self.__class__(*simplified_children)


class Arcsinh(SpecificFunction):
    """ Arcsinh function """

    def __init__(self, child):
        super().__init__(np.arcsinh, child)

    def _function_diff(self, children, idx):
        """ See :meth:`pybamm.Symbol._function_diff()`. """
        return 1 / Sqrt(children[0] ** 2 + 1)


def arcsinh(child):
    " Returns arcsinh function of child. "
    return pybamm.simplify_if_constant(Arcsinh(child), keep_domains=True)


class Cos(SpecificFunction):
    """ Cosine function """

    def __init__(self, child):
        super().__init__(np.cos, child)

    def _function_diff(self, children, idx):
        """ See :meth:`pybamm.Symbol._function_diff()`. """
        return -Sin(children[0])


def cos(child):
    " Returns cosine function of child. "
    return pybamm.simplify_if_constant(Cos(child), keep_domains=True)


class Cosh(SpecificFunction):
    """ Hyberbolic cosine function """

    def __init__(self, child):
        super().__init__(np.cosh, child)

    def _function_diff(self, children, idx):
        """ See :meth:`pybamm.Function._function_diff()`. """
        return Sinh(children[0])


def cosh(child):
    " Returns hyperbolic cosine function of child. "
    return pybamm.simplify_if_constant(Cosh(child), keep_domains=True)


class Exponential(SpecificFunction):
    """ Exponential function """

    def __init__(self, child):
        super().__init__(np.exp, child)

    def _function_diff(self, children, idx):
        """ See :meth:`pybamm.Function._function_diff()`. """
        return Exponential(children[0])


def exp(child):
    " Returns exponential function of child. "
    return pybamm.simplify_if_constant(Exponential(child), keep_domains=True)


class Log(SpecificFunction):
    """ Logarithmic function """

    def __init__(self, child):
        super().__init__(np.log, child)

    def _function_evaluate(self, evaluated_children):
        # don't raise RuntimeWarning for NaNs
        with np.errstate(invalid="ignore"):
            return np.log(*evaluated_children)

    def _function_diff(self, children, idx):
        """ See :meth:`pybamm.Function._function_diff()`. """
        return 1 / children[0]


def log(child, base="e"):
    " Returns logarithmic function of child (any base, default 'e'). "
    if base == "e":
        return pybamm.simplify_if_constant(Log(child), keep_domains=True)
    else:
        return Log(child) / np.log(base)


def log10(child):
    " Returns logarithmic function of child, with base 10. "
    return log(child, base=10)


def max(child):
    """
    Returns max function of child. Not to be confused with :meth:`pybamm.maximum`, which
    returns the larger of two objects.
    """
    return pybamm.simplify_if_constant(Function(np.max, child), keep_domains=True)


def min(child):
    """
    Returns min function of child. Not to be confused with :meth:`pybamm.minimum`, which
    returns the smaller of two objects.
    """
    return pybamm.simplify_if_constant(Function(np.min, child), keep_domains=True)


def sech(child):
    " Returns hyperbolic sec function of child. "
    return pybamm.simplify_if_constant(1 / Cosh(child), keep_domains=True)


class Sin(SpecificFunction):
    """ Sine function """

    def __init__(self, child):
        super().__init__(np.sin, child)

    def _function_diff(self, children, idx):
        """ See :meth:`pybamm.Function._function_diff()`. """
        return Cos(children[0])


def sin(child):
    " Returns sine function of child. "
    return pybamm.simplify_if_constant(Sin(child), keep_domains=True)


class Sinh(SpecificFunction):
    """ Hyperbolic sine function """

    def __init__(self, child):
        super().__init__(np.sinh, child)

    def _function_diff(self, children, idx):
        """ See :meth:`pybamm.Function._function_diff()`. """
        return Cosh(children[0])


def sinh(child):
    " Returns hyperbolic sine function of child. "
    return pybamm.simplify_if_constant(Sinh(child), keep_domains=True)


class Sqrt(SpecificFunction):
    """ Square root function """

    def __init__(self, child):
        super().__init__(np.sqrt, child)

    def _function_evaluate(self, evaluated_children):
        # don't raise RuntimeWarning for NaNs
        with np.errstate(invalid="ignore"):
            return np.sqrt(*evaluated_children)

    def _function_diff(self, children, idx):
        """ See :meth:`pybamm.Function._function_diff()`. """
        return 1 / (2 * Sqrt(children[0]))


def sqrt(child):
    " Returns square root function of child. "
    return pybamm.simplify_if_constant(Sqrt(child), keep_domains=True)


class Tanh(SpecificFunction):
    """ Hyperbolic tan function """

    def __init__(self, child):
        super().__init__(np.tanh, child)

    def _function_diff(self, children, idx):
        """ See :meth:`pybamm.Function._function_diff()`. """
        return sech(children[0]) ** 2


def tanh(child):
    " Returns hyperbolic tan function of child. "
    return pybamm.simplify_if_constant(Tanh(child), keep_domains=True)


class Arctan(SpecificFunction):
    """ Arctan function """

    def __init__(self, child):
        super().__init__(np.arctan, child)

    def _function_diff(self, children, idx):
        """ See :meth:`pybamm.Function._function_diff()`. """
        return 1 / (children[0] ** 2 + 1)


def arctan(child):
    " Returns hyperbolic tan function of child. "
    return pybamm.simplify_if_constant(Arctan(child), keep_domains=True)


