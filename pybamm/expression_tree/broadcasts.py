#
# Unary operator classes and methods
#
import numbers
import numpy as np
import pybamm


class Broadcast(pybamm.SpatialOperator):
    """A node in the expression tree representing a broadcasting operator.
    Broadcasts a child (which *must* have empty domain) to a specified domain. After
    discretisation, this will evaluate to an array of the right shape for the specified
    domain.

    Parameters
    ----------
    child : :class:`Symbol`
        child node
    domain : iterable of string
        the domain to broadcast the child to
    name : str
        name of the node
    broadcast_type : str, optional
        Whether to broadcast to the full domain (primary and secondary) or only in the
        primary direction. Default is "full".

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, broadcast_domain, broadcast_type="full", name=None):
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
        super().__init__(name, child, domain)

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

        # Secondary broadcast to current collector is acceptable
        if broadcast_type == "secondary":
            if broadcast_domain == "current collector":
                domain = child.domain
            else:
                raise pybamm.DomainError

        # Otherwise only some domains can be broadcast
        else:
            if child.domain not in [
                [],
                broadcast_domain,
                ["current collector"],
                ["negative particle"],
                ["positive particle"],
            ]:
                raise pybamm.DomainError(
                    """
                    Domain of a broadcasted child must be [], ['current collector'],
                    ["negative particle"], ["positive particle"] or same as
                    'broadcast_domain' ('{}'), but is '{}'
                    """.format(
                        broadcast_domain, child.domain
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

        return Broadcast(child, self.broadcast_domain, self.broadcast_type)

    def _unary_new_copy(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

        return Broadcast(child, self.broadcast_domain, self.broadcast_type)

    def evaluate_for_shape(self):
        """
        Returns a vector of NaNs to represent the shape of a Broadcast.
        See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`
        """
        child_eval = self.children[0].evaluate_for_shape()
        vec = pybamm.evaluate_for_shape_using_domain(self.domain)

        if self.broadcast_type == "primary":
            return np.outer(child_eval, vec).reshape(-1, 1)
        elif self.broadcast_type == "secondary":
            return np.outer(
                pybamm.evaluate_for_shape_using_domain(self.broadcast_domain),
                child_eval,
            ).reshape(-1, 1)
        elif self.broadcast_type == "full":
            return child_eval * vec


class PrimaryBroadcast(Broadcast):
    "A class for primary broadcasts"

    def __init__(self, child, broadcast_domain, name=None):
        super().__init__(child, broadcast_domain, "primary", name)


class SecondaryBroadcast(Broadcast):
    "A class for secondary broadcasts"

    def __init__(self, child, broadcast_domain, name=None):
        super().__init__(child, broadcast_domain, "secondary", name)


class FullBroadcast(Broadcast):
    "A class for full broadcasts"

    def __init__(self, child, broadcast_domain, name=None):
        super().__init__(child, broadcast_domain, "full", name)
