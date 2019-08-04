#
# Function classes and methods
#
import autograd
import numpy as np
import pybamm
from inspect import signature
from scipy.sparse import csr_matrix


class Function(pybamm.Symbol):
    """A node in the expression tree representing an arbitrary function

    Parameters
    ----------
    function : method
        A function can have 0 or many inputs. If no inputs are given, self.evaluate()
        simply returns func(). Otherwise, self.evaluate(t, y) returns
        func(child0.evaluate(t, y), child1.evaluate(t, y), etc).
    children : :class:`pybamm.Symbol`
        The children nodes to apply the function to

    **Extends:** :class:`pybamm.Symbol`
    """

    def __init__(self, function, *children):

        try:
            name = "function ({})".format(function.__name__)
        except AttributeError:
            name = "function ({})".format(function.__class__)
        children_list = list(children)
        domain = self.get_children_domains(children_list)
        auxiliary_domains = self.get_children_auxiliary_domains(children)

        self.function = function

        # hack to work out whether function takes any params
        # (signature doesn't work for numpy)
        if isinstance(function, np.ufunc):
            self.takes_no_params = False
        else:
            self.takes_no_params = len(signature(function).parameters) == 0

        super().__init__(
            name,
            children=children_list,
            domain=domain,
            auxiliary_domains=auxiliary_domains,
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

    def get_children_auxiliary_domains(self, children):
        "Combine auxiliary domains from children, at all levels"
        aux_domains = {}
        for child in children:
            for level in child.auxiliary_domains.keys():
                if (
                    not hasattr(aux_domains, level)
                    or aux_domains[level] == []
                    or child.auxiliary_domains[level] == aux_domains[level]
                ):
                    aux_domains[level] = child.auxiliary_domains[level]
                else:
                    raise pybamm.DomainError(
                        """children must have same or empty auxiliary domains,
                        not {!s} and {!s}""".format(
                            aux_domains[level], child.auxiliary_domains[level]
                        )
                    )

        return aux_domains

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            children = self.orphans
            partial_derivatives = [None] * len(children)
            for i, child in enumerate(self.children):
                # if variable appears in the function,use autograd to differentiate
                # function, and apply chain rule
                if variable.id in [symbol.id for symbol in child.pre_order()]:
                    partial_derivatives[i] = child.diff(variable) * self._diff(children)

            # remove None entries
            partial_derivatives = list(filter(None, partial_derivatives))

            derivative = sum(partial_derivatives)
            if derivative == 0:
                derivative = pybamm.Scalar(0)

            return derivative

    def _diff(self, children):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return Function(autograd.elementwise_grad(self.function), *children)

    def _jac(self, variable):
        """ See :meth:`pybamm.Symbol._jac()`. """

        if all(child.evaluates_to_number() for child in self.children):
            # if children all evaluate to numbers the return zeros
            # of right size
            jac = csr_matrix((1, variable.evaluation_array.count(True)))
            jacobian = pybamm.Matrix(jac)
        else:

            # if at least one child contains variable dependence, then
            # calculate the required partial jacobians and add them
            jacobian = None
            children = self.orphans
            for child in children:
                if not child.evaluates_to_number():
                    jac_fun = self._diff(children) * child.jac(variable)

                    jac_fun.domain = self.domain
                    if jacobian is None:
                        jacobian = jac_fun
                    else:
                        jacobian += jac_fun

        return jacobian

    def evaluate(self, t=None, y=None, known_evals=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        if known_evals is not None:
            if self.id not in known_evals:
                evaluated_children = [None] * len(self.children)
                for i, child in enumerate(self.children):
                    evaluated_children[i], known_evals = child.evaluate(
                        t, y, known_evals
                    )
                known_evals[self.id] = self._function_evaluate(evaluated_children)
            return known_evals[self.id], known_evals
        else:
            evaluated_children = [child.evaluate(t, y) for child in self.children]
            return self._function_evaluate(evaluated_children)

    def evaluate_for_shape(self):
        """
        Default behaviour: has same shape as all child
        See :meth:`pybamm.Symbol.evaluate_for_shape()`
        """
        evaluated_children = [child.evaluate_for_shape() for child in self.children]
        return self._function_evaluate(evaluated_children)

    def _function_evaluate(self, evaluated_children):
        if self.takes_no_params is True:
            return self.function()
        else:
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
        return pybamm.Function(self.function, *children)

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
        if self.takes_no_params is True:
            # If self.function() takes no parameters then we can always simplify it
            return pybamm.Scalar(self.function())
        elif isinstance(self.function, pybamm.GetConstantCurrent):
            # If self.function() is a constant current then simplify to scalar
            return pybamm.Scalar(self.function.parameters_eval["Current [A]"])
        else:
            return pybamm.Function(self.function, *simplified_children)


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


class Cos(SpecificFunction):
    """ Cosine function """

    def __init__(self, child):
        super().__init__(np.cos, child)

    def _diff(self, children):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return -Sin(children[0])


def cos(child):
    " Returns cosine function of child. "
    return Cos(child)


class Cosh(SpecificFunction):
    """ Hyberbolic cosine function """

    def __init__(self, child):
        super().__init__(np.cosh, child)

    def _diff(self, children):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return Sinh(children[0])


def cosh(child):
    " Returns hyperbolic cosine function of child. "
    return Cosh(child)


class Exponential(SpecificFunction):
    """ Exponential function """

    def __init__(self, child):
        super().__init__(np.exp, child)

    def _diff(self, children):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return Exponential(children[0])


def exp(child):
    " Returns exponential function of child. "
    return Exponential(child)


class Log(SpecificFunction):
    """ Logarithmic function """

    def __init__(self, child):
        super().__init__(np.log, child)

    def _diff(self, children):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return 1 / children[0]


def log(child):
    " Returns logarithmic function of child. "
    return Log(child)


def max(child):
    " Returns max function of child. "
    return Function(np.max, child)


def min(child):
    " Returns min function of child. "
    return Function(np.min, child)


class Sin(SpecificFunction):
    """ Sine function """

    def __init__(self, child):
        super().__init__(np.sin, child)

    def _diff(self, children):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return Cos(children[0])


def sin(child):
    " Returns sine function of child. "
    return Sin(child)


class Sinh(SpecificFunction):
    """ Hyperbolic sine function """

    def __init__(self, child):
        super().__init__(np.sinh, child)

    def _diff(self, children):
        """ See :meth:`pybamm.Symbol._diff()`. """
        return Cosh(children[0])


def sinh(child):
    " Returns hyperbolic sine function of child. "
    return Sinh(child)
