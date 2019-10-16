#
# Calculate the Jacobian of a symbol
#
import pybamm


class Jacobian(object):
    def __init__(self, known_jacs=None):
        self._known_jacs = known_jacs or {}

    def jac(self, symbol, variable):
        """
        This function recurses down the tree, computing the Jacobian using
        the Jacobians defined in classes derived from pybamm.Symbol. E.g. the
        Jacobian of a 'pybamm.Multiplication' is computed via the product rule.
        If the Jacobian of a symbol has already been calculated, the stored value
        is returned.
        Note: The Jacobian is the derivative of a symbol with respect to a (slice of)
        a State Vector.

        Parameters
        ----------
        symbol : :class:`pybamm.Symbol`
            The symbol to calculate the Jacobian of
        variable : :class:`pybamm.Symbol`
            The variable with respect to which to differentiate

        Returns
        -------
        :class:`pybamm.Symbol`
            Symbol representing the Jacobian
        """

        try:
            return self._known_jacs[symbol.id]
        except KeyError:
            jac = self._jac(symbol, variable)
            self._known_jacs[symbol.id] = jac

            return jac

    def _jac(self, symbol, variable):
        """ See :meth:`Jacobian.jac()`. """
        if variable.id == symbol.id:
            jac = pybamm.Scalar(1)
        else:
            jac = symbol._jac(variable)
        # jacobian removes the domain(s)
        jac.domain = []
        jac.auxiliary_domains = {}
        return jac
