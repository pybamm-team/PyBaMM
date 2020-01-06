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
        broadcast_auxiliary_domains=None,
        broadcast_type="full",
        name=None,
    ):
        # Convert child to scalar if it is a number
        if isinstance(child, numbers.Number):
            child = pybamm.Scalar(child)
        # Convert domain to list if it's a string
        if isinstance(broadcast_domain, str):
            broadcast_domain = [broadcast_domain]

        if name is None:
            name = "broadcast"

        # perform some basic checks and set attributes
        domain, auxiliary_domains = self.check_and_set_domains(
            child, broadcast_type, broadcast_domain, broadcast_auxiliary_domains
        )
        self.broadcast_type = broadcast_type
        self.broadcast_domain = broadcast_domain
        super().__init__(name, child, domain, auxiliary_domains)

    def check_and_set_domains(
        self, child, broadcast_type, broadcast_domain, broadcast_auxiliary_domains
    ):
        """
        Set broadcast domain and broadcast type, performing basic checks to make sure
        it is compatible with the child
        """
        raise NotImplementedError

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

    def check_and_set_domains(
        self, child, broadcast_type, broadcast_domain, broadcast_auxiliary_domains
    ):
        "See :meth:`Broadcast.check_and_set_domains`"
        # Can only do primary broadcast from current collector to electrode or from
        # electrode to particle
        if child.domain == []:
            pass
        elif child.domain[0] in [
            "negative electrode",
            "separator",
            "positive electrode",
        ] and broadcast_domain[0] not in ["negative particle", "positive particle"]:
            raise pybamm.DomainError(
                """Primary broadcast from electrode or separator must be to particle
                domains"""
            )
        elif child.domain[0] in ["negative particle", "positive particle"]:
            raise pybamm.DomainError("Cannot do primary broadcast from particle domain")

        domain = broadcast_domain
        if broadcast_auxiliary_domains is None:
            if child.domain != []:
                auxiliary_domains = {"secondary": child.domain}
            else:
                auxiliary_domains = {}
        else:
            auxiliary_domains = broadcast_auxiliary_domains

        return domain, auxiliary_domains

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


class SecondaryBroadcast(Broadcast):
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
        super().__init__(child, broadcast_domain, broadcast_type="secondary", name=name)

    def check_and_set_domains(
        self, child, broadcast_type, broadcast_domain, broadcast_auxiliary_domains
    ):
        "See :meth:`Broadcast.check_and_set_domains`"

        # Can only do secondary broadcast from particle to electrode or from
        # current collector to electrode
        if child.domain[0] in [
            "negative particle",
            "positive particle",
        ] and broadcast_domain[0] not in [
            "negative electrode",
            "separator",
            "positive electrode",
        ]:
            raise pybamm.DomainError(
                """Secondary broadcast from particle domain must be to electrode or
                separator domains"""
            )
        elif child.domain[0] in [
            "negative electrode",
            "separator",
            "positive electrode",
        ] and broadcast_domain != ["current collector"]:
            raise pybamm.DomainError(
                """Secondary broadcast from electrode or separator must be to
                current collector domains"""
            )
        elif child.domain == ["current collector"]:
            raise pybamm.DomainError(
                "Cannot do secondary broadcast from current collector domain"
            )
        # Domain stays the same as child domain and broadcast domain is secondary
        # domain
        domain = child.domain
        auxiliary_domains = {"secondary": broadcast_domain}
        # Child's secondary domain becomes tertiary domain
        if "secondary" in child.auxiliary_domains:
            auxiliary_domains["tertiary"] = child.auxiliary_domains["secondary"]

        return domain, auxiliary_domains

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """
        return SecondaryBroadcast(child, self.broadcast_domain)

    def _unary_new_copy(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """
        return SecondaryBroadcast(child, self.broadcast_domain)

    def evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Broadcast.
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(self.domain)
        return np.outer(vec, child_eval).reshape(-1, 1)


class FullBroadcast(Broadcast):
    "A class for full broadcasts"

    def __init__(self, child, broadcast_domain, auxiliary_domains, name=None):
        if auxiliary_domains == "current collector":
            auxiliary_domains = {"secondary": "current collector"}
        super().__init__(
            child,
            broadcast_domain,
            broadcast_auxiliary_domains=auxiliary_domains,
            broadcast_type="full",
            name=name,
        )

    def check_and_set_domains(
        self, child, broadcast_type, broadcast_domain, broadcast_auxiliary_domains
    ):
        "See :meth:`Broadcast.check_and_set_domains`"

        # Variables on the current collector can only be broadcast to 'primary'
        if child.domain == ["current collector"]:
            raise pybamm.DomainError(
                "Cannot do full broadcast from current collector domain"
            )
        domain = broadcast_domain
        if broadcast_auxiliary_domains is None:
            if child.domain != []:
                auxiliary_domains = {"secondary": child.domain}
            else:
                auxiliary_domains = {}
        else:
            auxiliary_domains = broadcast_auxiliary_domains

        return domain, auxiliary_domains

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


def ones_like(*symbols):
    """
    Create a symbol with the same shape as the input symbol and with constant value '1',
    using `FullBroadcast`.

    Parameters
    ----------
    symbols : :class:`Symbol`
        Symbols whose shape to copy
    """
    # Make a symbol that combines all the children, to get the right domain
    # that takes all the child symbols into account
    sum_symbol = symbols[0]
    for sym in symbols:
        sum_symbol += sym

    # Just return scalar 1 if symbol has no domain (no broadcasting necessary)
    if sum_symbol.domain == []:
        return pybamm.Scalar(1)
    else:
        return FullBroadcast(1, sum_symbol.domain, sum_symbol.auxiliary_domains)
