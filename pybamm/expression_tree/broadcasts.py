#
# Unary operator classes and methods
#
import numbers
import numpy as np
import pybamm


class Broadcast(pybamm.SpatialOperator):
    """A node in the expression tree representing a broadcasting operator.
    Broadcasts a child to a specified domain. After discretisation, this will evaluate
    to an array of the right shape for the specified domain.

    Parameters
    ----------
    child : :class:`Symbol`
        child node
    broadcast_domain : iterable of str
        Primary domain for broadcast. This will become the domain of the symbol
    auxiliary_domain : iterable of str
        Secondary domain for broadcast. Currently, this is only used for testing that
        symbols have the right shape.
    broadcast_type : str, optional
        Whether to broadcast to the full domain (primary and secondary) or only in the
        primary direction. Default is "full".
    name : str
        name of the node

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(
        self,
        child,
        broadcast_domain,
        auxiliary_domains=None,
        broadcast_type="full",
        name=None,
    ):
        # Convert child to scalar if it is a number
        if isinstance(child, numbers.Number):
            child = pybamm.Scalar(child)

        if name is None:
            name = "broadcast"

        # perform some basic checks and set attributes
        domain = self.check_and_set_domain_and_broadcast_type(
            child, broadcast_domain, broadcast_type
        )
        self.broadcast_type = broadcast_type
        self.broadcast_domain = broadcast_domain
        if auxiliary_domains is None:
            if child.domain != []:
                auxiliary_domains = {"secondary": child.domain}
            else:
                auxiliary_domains = {}
        super().__init__(name, child, domain, auxiliary_domains)

    def check_and_set_domain_and_broadcast_type(
        self, child, broadcast_domain, broadcast_type
    ):
        """
        Set broadcast domain and broadcast type, performing basic checks to make sure
        it is compatible with the child
        """
        # Acceptable broadcast types
        if broadcast_type not in ["primary", "secondary", "full"]:
            raise KeyError(
                """Broadcast type must be either: 'primary', 'secondary', or 'full' and
                not {}""".format(
                    broadcast_type
                )
            )

        domain = broadcast_domain

        # Variables on the current collector can only be broadcast to 'primary'
        if broadcast_type == "full":
            if child.domain == ["current collector"]:
                raise ValueError(
                    """
                    Variables on the current collector must be broadcast to 'primary'
                    only
                    """
                )
        return domain

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

        return Broadcast(
            child, self.broadcast_domain, self.auxiliary_domains, self.broadcast_type
        )

    def _unary_new_copy(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

        return Broadcast(
            child, self.broadcast_domain, self.auxiliary_domains, self.broadcast_type
        )

    def evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Broadcast.
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(self.domain)

        if self.broadcast_type == "primary":
            return np.outer(child_eval, vec).reshape(-1, 1)
        elif self.broadcast_type == "full":
            return child_eval * vec


class PrimaryBroadcast(Broadcast):
    """A node in the expression tree representing a primary broadcasting operator.
    Broadcasts in a `primary` dimension only. That is, makes explicit copies

    Parameters
    ----------
    child : :class:`Symbol`
        child node
    broadcast_domain : iterable of str
        Primary domain for broadcast. This will become the domain of the symbol
    name : str
        name of the node

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, broadcast_domain, name=None):
        super().__init__(child, broadcast_domain, broadcast_type="primary", name=name)

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """
        return PrimaryBroadcast(child, self.broadcast_domain)

    def _unary_new_copy(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """
        return PrimaryBroadcast(child, self.broadcast_domain)

    def evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Broadcast.
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(self.domain)
        return np.outer(child_eval, vec).reshape(-1, 1)


class FullBroadcast(Broadcast):
    "A class for full broadcasts"

    def __init__(self, child, broadcast_domain, auxiliary_domains, name=None):
        if auxiliary_domains == "current collector":
            auxiliary_domains = {"secondary": "current collector"}
        super().__init__(
            child,
            broadcast_domain,
            auxiliary_domains=auxiliary_domains,
            broadcast_type="full",
            name=name,
        )

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """
        return FullBroadcast(child, self.broadcast_domain, self.auxiliary_domains)

    def _unary_new_copy(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """
        return FullBroadcast(child, self.broadcast_domain, self.auxiliary_domains)

    def evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Broadcast.
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(
            self.domain, self.auxiliary_domains
        )

        return child_eval * vec
