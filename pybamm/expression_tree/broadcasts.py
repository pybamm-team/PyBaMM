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

    For an example of broadcasts in action, see
    `this example notebook
    <https://github.com/pybamm-team/PyBaMM/blob/master/examples/notebooks/expression_tree/broadcasts.ipynb>`_

    Parameters
    ----------
    child : :class:`Symbol`
        child node
    broadcast_domain : iterable of str
        Primary domain for broadcast. This will become the domain of the symbol
    broadcast_auxiliary_domains : dict of str
        Auxiliary domains for broadcast.
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
        broadcast_type="full to nodes",
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

    def _unary_simplify(self, simplified_child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """
        return self._unary_new_copy(simplified_child)


class PrimaryBroadcast(Broadcast):
    """A node in the expression tree representing a primary broadcasting operator.
    Broadcasts in a `primary` dimension only. That is, makes explicit copies of the
    symbol in the domain specified by `broadcast_domain`. This should be used for
    broadcasting from a "larger" scale to a "smaller" scale, for example broadcasting
    temperature T(x) from the electrode to the particles, or broadcasting current
    collector current i(y, z) from the current collector to the electrodes.

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
        super().__init__(
            child, broadcast_domain, broadcast_type="primary to nodes", name=name
        )

    def check_and_set_domains(
        self, child, broadcast_type, broadcast_domain, broadcast_auxiliary_domains
    ):
        "See :meth:`Broadcast.check_and_set_domains`"
        # Can only do primary broadcast from current collector to electrode or particle
        # or from electrode to particle. Note current collector to particle *is* allowed
        if child.domain == []:
            pass
        elif child.domain == ["current collector"] and broadcast_domain[0] not in [
            "negative electrode",
            "separator",
            "positive electrode",
            "negative particle",
            "positive particle",
        ]:
            raise pybamm.DomainError(
                """Primary broadcast from current collector domain must be to electrode
                or separator or particle domains"""
            )
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
        auxiliary_domains = {}
        if child.domain != []:
            auxiliary_domains["secondary"] = child.domain
        if "secondary" in child.auxiliary_domains:
            auxiliary_domains["tertiary"] = child.auxiliary_domains["secondary"]

        return domain, auxiliary_domains

    def _unary_new_copy(self, child):
        """ See :meth:`pybamm.UnaryOperator._unary_new_copy()`. """
        return self.__class__(child, self.broadcast_domain)

    def _evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Broadcast.
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(self.domain)
        return np.outer(child_eval, vec).reshape(-1, 1)


class PrimaryBroadcastToEdges(PrimaryBroadcast):
    "A primary broadcast onto the edges of the domain"

    def __init__(self, child, broadcast_domain, name=None):
        name = name or "broadcast to edges"
        super().__init__(child, broadcast_domain, name)
        self.broadcast_type = "primary to edges"

    def evaluates_on_edges(self):
        return True


class SecondaryBroadcast(Broadcast):
    """A node in the expression tree representing a primary broadcasting operator.
    Broadcasts in a `secondary` dimension only. That is, makes explicit copies of the
    symbol in the domain specified by `broadcast_domain`. This should be used for
    broadcasting from a "smaller" scale to a "larger" scale, for example broadcasting
    SPM particle concentrations c_s(r) from the particles to the electrodes. Note that
    this wouldn't be used to broadcast particle concentrations in the DFN, since these
    already depend on both x and r.

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
        super().__init__(
            child, broadcast_domain, broadcast_type="secondary to nodes", name=name
        )

    def check_and_set_domains(
        self, child, broadcast_type, broadcast_domain, broadcast_auxiliary_domains
    ):
        "See :meth:`Broadcast.check_and_set_domains`"

        # Can only do secondary broadcast from particle to electrode or from
        # electrode to current collector
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

    def _unary_new_copy(self, child):
        """ See :meth:`pybamm.UnaryOperator._unary_new_copy()`. """
        return SecondaryBroadcast(child, self.broadcast_domain)

    def _evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Broadcast.
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(self.domain)
        return np.outer(vec, child_eval).reshape(-1, 1)


class SecondaryBroadcastToEdges(SecondaryBroadcast):
    "A secondary broadcast onto the edges of a domain"

    def __init__(self, child, broadcast_domain, name=None):
        name = name or "broadcast to edges"
        super().__init__(child, broadcast_domain, name)
        self.broadcast_type = "secondary to edges"

    def evaluates_on_edges(self):
        return True


class FullBroadcast(Broadcast):
    "A class for full broadcasts"

    def __init__(self, child, broadcast_domain, auxiliary_domains, name=None):
        if isinstance(auxiliary_domains, str):
            auxiliary_domains = {"secondary": auxiliary_domains}
        super().__init__(
            child,
            broadcast_domain,
            broadcast_auxiliary_domains=auxiliary_domains,
            broadcast_type="full to nodes",
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
        auxiliary_domains = broadcast_auxiliary_domains or {}

        return domain, auxiliary_domains

    def _unary_new_copy(self, child):
        """ See :meth:`pybamm.UnaryOperator._unary_new_copy()`. """
        return FullBroadcast(child, self.broadcast_domain, self.auxiliary_domains)

    def _evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Broadcast.
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(
            self.domain, self.auxiliary_domains
        )

        return child_eval * vec


class FullBroadcastToEdges(FullBroadcast):
    """
    A full broadcast onto the edges of a domain (edges of primary dimension, nodes of
    other dimensions)
    """

    def __init__(self, child, broadcast_domain, auxiliary_domains, name=None):
        name = name or "broadcast to edges"
        super().__init__(child, broadcast_domain, auxiliary_domains, name)
        self.broadcast_type = "full to edges"

    def evaluates_on_edges(self):
        return True


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
