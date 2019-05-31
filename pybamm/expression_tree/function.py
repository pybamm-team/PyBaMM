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

    **Extends:** :class:`UnaryOperator`
    """

    def __init__(self, function, *children):

        name = "function ({})".format(function.__name__)
        children_list = list(children)
        domain = self.get_children_domains(children_list)
        super().__init__(name, children=children_list, domain=domain)

        self.function = function
        self.number_of_parameters = len(children_list)

        # hack to work out whether function takes any params
        # (signature doesn't work for numpy)
        # if isinstance(func, np.ufunc):
        #     self.takes_no_params = False
        # else:
        #     self.takes_no_params = len(signature(func).parameters) == 0

    def get_children_domains(self, children):
        """Obtains the unique domain of the children. If the
        children have different domains then raise an error"""

        # TODO: maybe relax this for domain=[]
        domains = [None] * len(children)
        for i, child in enumerate(children):
            domains[i] = child.domain

        distinct_domains = set(domains)

        if len(distinct_domains) > 1:
            raise pybamm.DomainError(
                "Functions can only be applied to variables on the same domain"
            )

        return domains[0]

    def diff(self, variable):
        """ See :meth:`pybamm.Symbol.diff()`. """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            children = self.orphans
            partial_derivatives = [None] * len(children)
            for i, child in enumerate(children):
                # if variable appears in the function,use autograd to differentiate
                # function, and apply chain rule
                if variable.id in [symbol.id for symbol in child.pre_order()]:
                    partial_derivatives[i] = child.diff(variable) * Function(
                        autograd.grad(self.func), *children
                    )

            # remove None entries
            partial_derivatives = list(filter(None), partial_derivatives)

            derivative = sum(partial_derivatives)
            if derivative == 0:
                derivative = pybamm.Scalar(0)

            return derivative

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """

        children = self.orphans

        if all(child.evaluates_to_number() for child in children):
            # if children all evaluate to numbers the return zeros
            # of right size
            variable_y_indices = np.arange(
                variable.y_slice.start, variable.y_slice.stop
            )
            jac = csr_matrix((1, np.size(variable_y_indices)))
            jacobian = pybamm.Matrix(jac)
        else:

            # if at least one child contains variable dependence, then
            # calculate the required partial jacobians and add them
            jacobian = None
            for child in children:
                if not child.evaluates_to_number():
                    jac_fun = Function(
                        autograd.elementwise_grad(self.func), *children
                    ) * child.jac(variable)

                    jac_fun.domain = self.domain

                    if jacobian is None:
                        jacobian = jac_fun
                    else:
                        jacobian += jac_fun

        return jacobian

    def _unary_evaluate(self, child):
        """ See :meth:`UnaryOperator._unary_evaluate()`. """
        if self.takes_no_params:
            return self.func()
        else:
            return self.func(child)

    def _unary_new_copy(self, child):
        """ See :meth:`UnaryOperator._unary_new_copy()`. """
        return pybamm.Function(self.func, child)

    def _unary_simplify(self, simplified_child):
        """ See :meth:`UnaryOperator._unary_simplify()`. """
        if self.takes_no_params:
            # If self.func() takes no parameters then we can always simplify it
            return pybamm.Scalar(self.func())
        else:
            return pybamm.Function(self.func, simplified_child)

