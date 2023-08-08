#
# Unary operator classes and methods
#
import numbers

import numpy as np
from scipy.sparse import csr_matrix

import pybamm


class Broadcast(pybamm.SpatialOperator):
    """
    A node in the expression tree representing a broadcasting operator.
    Broadcasts a child to a specified domain. After discretisation, this will evaluate
    to an array of the right shape for the specified domain.

    For an example of broadcasts in action, see
    `this example notebook
    <https://github.com/pybamm-team/PyBaMM/blob/develop/docs/source/examples/notebooks/expression_tree/broadcasts.ipynb>`_

    Parameters
    ----------
    child : :class:`Symbol`
        child node
    domains : iterable of str
        Domain(s) of the symbol after broadcasting
    name : str
        name of the node
    """

    def __init__(self, child, domains, name=None):
        if name is None:
            name = "broadcast"
        super().__init__(name, child, domains=domains)

    @property
    def broadcasts_to_nodes(self):
        if self.broadcast_type.endswith("nodes"):
            return True
        else:
            return False

    def _sympy_operator(self, child):
        """Override :meth:`pybamm.UnaryOperator._sympy_operator`"""
        return child


class PrimaryBroadcast(Broadcast):
    """
    A node in the expression tree representing a primary broadcasting operator.
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
    """

    def __init__(self, child, broadcast_domain, name=None):
        # Convert child to scalar if it is a number
        if isinstance(child, numbers.Number):
            child = pybamm.Scalar(child)
        # Convert domain to list if it's a string
        if isinstance(broadcast_domain, str):
            broadcast_domain = [broadcast_domain]
        # perform some basic checks and set attributes
        domains = self.check_and_set_domains(child, broadcast_domain)
        self.broadcast_domain = broadcast_domain
        self.broadcast_type = "primary to nodes"
        super().__init__(child, domains, name=name)

    def check_and_set_domains(self, child, broadcast_domain):
        """See :meth:`Broadcast.check_and_set_domains`"""
        # Can only do primary broadcast from current collector to electrode,
        # particle-size or particle or from electrode to particle-size or particle.
        # Note e.g. current collector to particle *is* allowed
        if broadcast_domain == []:
            raise pybamm.DomainError("Cannot Broadcast an object into empty domain.")
        if child.domain == []:
            pass
        elif child.domain == ["current collector"] and not (
            broadcast_domain[0]
            in [
                "negative electrode",
                "separator",
                "positive electrode",
            ]
            or "particle" in broadcast_domain[0]
        ):
            raise pybamm.DomainError(
                """Primary broadcast from current collector domain must be to electrode
                or separator or particle or particle size domains"""
            )
        elif (
            child.domain[0]
            in [
                "negative electrode",
                "separator",
                "positive electrode",
            ]
            and "particle" not in broadcast_domain[0]
        ):
            raise pybamm.DomainError(
                """Primary broadcast from electrode or separator must be to particle
                or particle size domains"""
            )
        elif child.domain[0] in [
            "negative particle size",
            "positive particle size",
        ] and broadcast_domain[0] not in ["negative particle", "positive particle"]:
            raise pybamm.DomainError(
                """Primary broadcast from particle size domain must be to particle
                domain"""
            )
        elif child.domain[0] in ["negative particle", "positive particle"]:
            raise pybamm.DomainError("Cannot do primary broadcast from particle domain")

        domains = {
            "primary": broadcast_domain,
            "secondary": child.domain,
            "tertiary": child.domains["secondary"],
            "quaternary": child.domains["tertiary"],
        }

        return domains

    def _unary_new_copy(self, child):
        """See :meth:`pybamm.UnaryOperator._unary_new_copy()`."""
        return self.__class__(child, self.broadcast_domain)

    def _evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Broadcast.
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(self.domains["primary"])
        return np.outer(child_eval, vec).reshape(-1, 1)

    def reduce_one_dimension(self):
        """Reduce the broadcast by one dimension."""
        return self.orphans[0]


class PrimaryBroadcastToEdges(PrimaryBroadcast):
    """A primary broadcast onto the edges of the domain."""

    def __init__(self, child, broadcast_domain, name=None):
        name = name or "broadcast to edges"
        super().__init__(child, broadcast_domain, name)
        self.broadcast_type = "primary to edges"

    def _evaluates_on_edges(self, dimension):
        return True


class SecondaryBroadcast(Broadcast):
    """
    A node in the expression tree representing a secondary broadcasting operator.
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
        Secondary domain for broadcast. This will become the secondary domain of the
        symbol, shifting the child's `secondary` and `tertiary` (if present) over by
        one position.
    name : str
        name of the node
    """

    def __init__(self, child, broadcast_domain, name=None):
        # Convert domain to list if it's a string
        if isinstance(broadcast_domain, str):
            broadcast_domain = [broadcast_domain]
        # perform some basic checks and set attributes
        domains = self.check_and_set_domains(child, broadcast_domain)
        self.broadcast_domain = broadcast_domain
        self.broadcast_type = "secondary to nodes"
        super().__init__(child, domains, name=name)

    def check_and_set_domains(self, child, broadcast_domain):
        """See :meth:`Broadcast.check_and_set_domains`"""
        if child.domain == []:
            raise TypeError(
                "Cannot take SecondaryBroadcast of an object with empty domain. "
                "Use PrimaryBroadcast instead."
            )
        # Can only do secondary broadcast from particle to electrode or current
        # collector or from electrode to current collector
        if child.domain[0] in [
            "negative particle",
            "positive particle",
        ] and broadcast_domain[0] not in [
            "negative particle size",
            "positive particle size",
            "negative electrode",
            "separator",
            "positive electrode",
            "current collector",
        ]:
            raise pybamm.DomainError(
                """Secondary broadcast from particle domain must be to particle-size,
                electrode, separator, or current collector domains"""
            )
        if child.domain[0] in [
            "negative particle size",
            "positive particle size",
        ] and broadcast_domain[0] not in [
            "negative electrode",
            "separator",
            "positive electrode",
            "current collector",
        ]:
            raise pybamm.DomainError(
                """Secondary broadcast from particle size domain must be to
                electrode or separator or current collector domains"""
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
        # Child's secondary domain becomes tertiary domain, tertiary becomes quaternary
        domains = {
            "primary": child.domains["primary"],
            "secondary": broadcast_domain,
            "tertiary": child.domains["secondary"],
            "quaternary": child.domains["tertiary"],
        }

        return domains

    def _unary_new_copy(self, child):
        """See :meth:`pybamm.UnaryOperator._unary_new_copy()`."""
        return SecondaryBroadcast(child, self.broadcast_domain)

    def _evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Broadcast.
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(self.domains["secondary"])
        return np.outer(vec, child_eval).reshape(-1, 1)

    def reduce_one_dimension(self):
        """Reduce the broadcast by one dimension."""
        return self.orphans[0]


class SecondaryBroadcastToEdges(SecondaryBroadcast):
    """A secondary broadcast onto the edges of a domain."""

    def __init__(self, child, broadcast_domain, name=None):
        name = name or "broadcast to edges"
        super().__init__(child, broadcast_domain, name)
        self.broadcast_type = "secondary to edges"

    def _evaluates_on_edges(self, dimension):
        return True


class TertiaryBroadcast(Broadcast):
    """
    A node in the expression tree representing a tertiary broadcasting operator.
    Broadcasts in a `tertiary` dimension only. That is, makes explicit copies of the
    symbol in the domain specified by `broadcast_domain`. This is used, e.g., for
    broadcasting particle concentrations c_s(r,R) in the MPM, which have a `primary`
    and `secondary` domain, to the electrode x, which is added as a `tertiary`
    domain. Note: the symbol for broadcast must already have a non-empty `secondary`
    domain.

    Parameters
    ----------
    child : :class:`Symbol`
        child node
    broadcast_domain : iterable of str
        The domain for broadcast. This will become the tertiary domain of the symbol.
        The `tertiary` domain of the child, if present, is shifted by one to the
        `quaternary` domain of the symbol.
    name : str
        name of the node
    """

    def __init__(self, child, broadcast_domain, name=None):
        # Convert domain to list if it's a string
        if isinstance(broadcast_domain, str):
            broadcast_domain = [broadcast_domain]
        # perform some basic checks and set attributes
        domains = self.check_and_set_domains(child, broadcast_domain)
        self.broadcast_domain = broadcast_domain
        self.broadcast_type = "tertiary to nodes"
        super().__init__(child, domains, name=name)

    def check_and_set_domains(self, child, broadcast_domain):
        """See :meth:`Broadcast.check_and_set_domains`"""
        if child.domains["secondary"] == []:
            raise TypeError(
                """Cannot take TertiaryBroadcast of an object without a secondary
                domain. Use SecondaryBroadcast instead."""
            )
        # Can only do tertiary broadcast to a "higher dimension" than the
        # secondary domain of child
        if child.domains["secondary"][0] in [
            "negative particle size",
            "positive particle size",
        ] and broadcast_domain[0] not in [
            "negative electrode",
            "separator",
            "positive electrode",
            "current collector",
        ]:
            raise pybamm.DomainError(
                """Tertiary broadcast from a symbol with particle size secondary
                domain must be to electrode, separator or current collector"""
            )
        if child.domains["secondary"][0] in [
            "negative electrode",
            "separator",
            "positive electrode",
        ] and broadcast_domain != ["current collector"]:
            raise pybamm.DomainError(
                """Tertiary broadcast from a symbol with an electrode or
                separator secondary domain must be to current collector"""
            )
        if child.domains["secondary"] == ["current collector"]:
            raise pybamm.DomainError(
                """Cannot do tertiary broadcast for symbol with a current collector
                secondary domain"""
            )
        # Primary and secondary domains stay the same as child's,
        # and broadcast domain is tertiary
        domains = {
            "primary": child.domains["primary"],
            "secondary": child.domains["secondary"],
            "tertiary": broadcast_domain,
            "quaternary": child.domains["tertiary"],
        }

        return domains

    def _unary_new_copy(self, child):
        """See :meth:`pybamm.UnaryOperator._unary_new_copy()`."""
        return self.__class__(child, self.broadcast_domain)

    def _evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Broadcast.
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(self.domains["tertiary"])
        return np.outer(vec, child_eval).reshape(-1, 1)

    def reduce_one_dimension(self):
        """Reduce the broadcast by one dimension."""
        raise NotImplementedError


class TertiaryBroadcastToEdges(TertiaryBroadcast):
    """A tertiary broadcast onto the edges of a domain."""

    def __init__(self, child, broadcast_domain, name=None):
        name = name or "broadcast to edges"
        super().__init__(child, broadcast_domain, name)
        self.broadcast_type = "tertiary to edges"

    def _evaluates_on_edges(self, dimension):
        return True


class FullBroadcast(Broadcast):
    """A class for full broadcasts."""

    def __init__(
        self,
        child,
        broadcast_domain=None,
        auxiliary_domains=None,
        broadcast_domains=None,
        name=None,
    ):
        # Convert child to scalar if it is a number
        if isinstance(child, numbers.Number):
            child = pybamm.Scalar(child)

        if isinstance(auxiliary_domains, str):
            auxiliary_domains = {"secondary": auxiliary_domains}
        broadcast_domains = self.read_domain_or_domains(
            broadcast_domain, auxiliary_domains, broadcast_domains
        )
        # perform some basic checks and set attributes
        domains = self.check_and_set_domains(child, broadcast_domains)
        self.broadcast_domain = broadcast_domains["primary"]
        self.broadcast_type = "full to nodes"
        super().__init__(child, domains, name=name)

    def check_and_set_domains(self, child, broadcast_domains):
        """See :meth:`Broadcast.check_and_set_domains`"""
        if broadcast_domains["primary"] == []:
            raise pybamm.DomainError(
                """Cannot do full broadcast to an empty primary domain"""
            )
        # Variables on the current collector can only be broadcast to 'primary'
        if child.domain == ["current collector"]:
            raise pybamm.DomainError(
                "Cannot do full broadcast from current collector domain"
            )

        return broadcast_domains

    def _unary_new_copy(self, child):
        """See :meth:`pybamm.UnaryOperator._unary_new_copy()`."""
        return self.__class__(child, broadcast_domains=self.domains)

    def _evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Broadcast.
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(self.domains)

        return child_eval * vec

    def reduce_one_dimension(self):
        """Reduce the broadcast by one dimension."""
        if self.domains["secondary"] == []:
            return self.orphans[0]
        elif self.domains["tertiary"] == []:
            return PrimaryBroadcast(self.orphans[0], self.domains["secondary"])
        else:
            domains = {
                "primary": self.domains["secondary"],
                "secondary": self.domains["tertiary"],
                "tertiary": self.domains["quaternary"],
            }
            return FullBroadcast(self.orphans[0], broadcast_domains=domains)


class FullBroadcastToEdges(FullBroadcast):
    """
    A full broadcast onto the edges of a domain (edges of primary dimension, nodes of
    other dimensions)
    """

    def __init__(
        self,
        child,
        broadcast_domain=None,
        auxiliary_domains=None,
        broadcast_domains=None,
        name=None,
    ):
        name = name or "broadcast to edges"
        super().__init__(
            child, broadcast_domain, auxiliary_domains, broadcast_domains, name
        )
        self.broadcast_type = "full to edges"

    def _evaluates_on_edges(self, dimension):
        return True

    def reduce_one_dimension(self):
        """Reduce the broadcast by one dimension."""
        if self.domains["secondary"] == []:
            return self.orphans[0]
        elif self.domains["tertiary"] == []:
            return PrimaryBroadcastToEdges(self.orphans[0], self.domains["secondary"])
        else:
            return FullBroadcastToEdges(
                self.orphans[0],
                broadcast_domains={
                    "primary": self.domains["secondary"],
                    "secondary": self.domains["tertiary"],
                },
            )


def full_like(symbols, fill_value):
    """
    Returns an array with the same shape and domains as the sum of the
    input symbols, with a constant value given by `fill_value`.

    Parameters
    ----------
    symbols : :class:`Symbol`
        Symbols whose shape to copy
    fill_value : number
        Value to assign
    """
    # Make a symbol that combines all the children, to get the right domain
    # that takes all the child symbols into account
    sum_symbol = symbols[0]
    for sym in symbols[1:]:
        sum_symbol += sym

    # Just return scalar if symbol shape is scalar
    if sum_symbol.evaluates_to_number():
        return pybamm.Scalar(fill_value)
    try:
        shape = sum_symbol.shape
        # use vector or matrix
        if shape[1] == 1:
            array_type = pybamm.Vector
        else:
            array_type = pybamm.Matrix
        # return dense array, except for a matrix of zeros
        if shape[1] != 1 and fill_value == 0:
            entries = csr_matrix(shape)
        else:
            entries = fill_value * np.ones(shape)

        return array_type(entries, domains=sum_symbol.domains)

    except NotImplementedError:
        if sum_symbol.shape_for_testing == (1, 1) or sum_symbol.shape_for_testing == (
            1,
        ):
            return pybamm.Scalar(fill_value)
        if sum_symbol.evaluates_on_edges("primary"):
            return FullBroadcastToEdges(
                fill_value, broadcast_domains=sum_symbol.domains
            )
        else:
            return FullBroadcast(fill_value, broadcast_domains=sum_symbol.domains)


def zeros_like(*symbols):
    """
    Returns an array with the same shape and domains as the sum of the
    input symbols, with each entry equal to zero.

    Parameters
    ----------
    symbols : :class:`Symbol`
        Symbols whose shape to copy
    """
    return full_like(symbols, 0)


def ones_like(*symbols):
    """
    Returns an array with the same shape and domains as the sum of the
    input symbols, with each entry equal to one.

    Parameters
    ----------
    symbols : :class:`Symbol`
        Symbols whose shape to copy
    """
    return full_like(symbols, 1)
