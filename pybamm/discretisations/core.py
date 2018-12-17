#
# Interface for discretisation
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class BaseDiscretisation(object):
    def __init__(self, mesh):
        self._mesh = mesh

    @property
    def mesh(self):
        return self._mesh

    def gradient(self, symbol, domain, y_slices, boundary_conditions):
        """How to discretise gradient operators"""
        raise NotImplementedError

    def divergence(self, symbol, domain, y_slices, boundary_conditions):
        """How to discretise divergence operators"""
        raise NotImplementedError

    def discretise_model(self, model):
        """Discretise a model.
        Currently inplace, could be changed to return a new model.

        Parameters
        ----------
        model : dict
            Model ({variable: equation} dict) to dicretise

        Returns
        -------
        y0 : :class:`numpy.array`
            Vector of initial conditions

        """
        # Set the y split for variables
        y_slices = self.get_variable_slices(model.keys())

        # Discretise and concatenate initial conditions, passing domain from variable
        for variable, equation in model.initial_conditions.items():
            model.initial_conditions[variable] = self.discretise_symbol(
                equation, variable.domain, y_slices, model.boundary_conditions
            )

        # Concatenate and evaluate initial conditions
        y0 = pybamm.Concatenation(*model.initial_conditions.values()).evaluate(None)

        # Discretise right-hand sides, passing domain from variable
        for variable, equation in model.rhs.items():
            model.rhs[variable] = self.discretise_symbol(
                equation, variable.domain, y_slices, model.boundary_conditions
            )

        # Concatenate and evaluate right-hand sides
        self._concatenated_rhs = pybamm.Concatenation(*model.rhs.values())

        def dydt(y):
            return self._concatenated_rhs.evaluate(y)

        return y0, dydt

    def get_variable_slices(self, variables):
        """Set the slicing for variables.

        Parameters
        ----------
        variables : dict_keys object containing Variable instances
            Variables for which to set the slices

        Returns
        -------
        y_slices : dict of {Variable: slice}
            The slices to take when solving (assigning chunks of y to each vector)
        """
        y_slices = {variable: None for variable in variables}
        start = 0
        for variable in variables:
            end = start + getattr(self.mesh, variable.domain).npts
            y_slices[variable] = slice(start, end)
            start = end

        return y_slices

    def discretise_symbol(self, symbol, domain, y_slices, boundary_conditions):
        """Discretise operators in model equations.

        Parameters
        ----------
        symbol : :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Symbol to discretise
        y_slices : dict of {Variable: slice}
            The slices to take when solving (assigning chunks of y to each vector)

        Returns
        -------
        :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Discretised symbol

        """
        if isinstance(symbol, pybamm.Gradient):
            return self.gradient(symbol.child, domain, y_slices, boundary_conditions)

        if isinstance(symbol, pybamm.Divergence):
            return self.divergence(symbol.child, domain, y_slices, boundary_conditions)

        elif isinstance(symbol, pybamm.BinaryOperator):
            new_left = self.discretise_symbol(
                symbol.left, domain, y_slices, boundary_conditions
            )
            new_right = self.discretise_symbol(
                symbol.right, domain, y_slices, boundary_conditions
            )
            return symbol.__class__(symbol.name, new_left, new_right)

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child = self.discretise_symbol(
                symbol.child, domain, y_slices, boundary_conditions
            )
            return symbol.__class__(symbol.name, new_child)

        elif isinstance(symbol, pybamm.Variable):
            return pybamm.Vector(y_slices[symbol])

        elif isinstance(symbol, pybamm.Scalar):
            return pybamm.Scalar(symbol.value)

        else:
            raise TypeError("""Cannot discretise {!r}""".format(symbol))

    def concatenate(self, *symbols):
        return pybamm.NumpyConcatenation(*symbols)


class MatrixVectorDiscretisation(BaseDiscretisation):
    """
    A base class for discretisations where the gradient and divergence operators are
    always matrix-vector multiplications.
    """

    def __init__(self, mesh):
        super().__init__(mesh)

    def gradient(self, symbol, domain, y_slices, boundary_conditions):
        """

        Parameters
        ----------
        symbol : :class:`pybamm.expression_tree.symbol.Symbol` (or subclass) instance
            Not necessarily a Variable (e.g. N = -D(c) * grad(c) is a Multiplication)
        """
        discretised_symbol = self.discretise_symbol(
            symbol, domain, y_slices, boundary_conditions
        )
        # Add boundary conditions if defined
        if symbol in boundary_conditions:
            lbc, rbc = boundary_conditions[symbol]
            discretised_symbol = self.concatenate(lbc, discretised_symbol, rbc)
        gradient_matrix = self.gradient_matrix(domain)
        return gradient_matrix * discretised_symbol

    def gradient_matrix(self, domain):
        raise NotImplementedError

    def divergence(self, symbol, domain, y_slices, boundary_conditions):
        discretised_symbol = self.discretise_symbol(
            symbol, domain, y_slices, boundary_conditions
        )
        # Add boundary conditions if defined
        if symbol in boundary_conditions:
            lbc, rbc = boundary_conditions[symbol]
            discretised_symbol = self.concatenate(lbc, discretised_symbol, rbc)
        divergence_matrix = self.gradient_matrix(domain)
        return divergence_matrix * discretised_symbol

    def divergence_matrix(self, domain):
        raise NotImplementedError
