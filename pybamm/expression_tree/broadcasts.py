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

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, domain, name=None, broadcast_type="full"):
        # Convert child to scalar if it is a number
        if isinstance(child, numbers.Number):
            child = pybamm.Scalar(child)

        # Check domain
        if child.domain not in [
            [],
            domain,
            ["current collector"],
            ["negative particle"],
            ["positive particle"],
        ]:
            raise pybamm.DomainError(
                """
                Domain of a broadcasted child must be [], ['current collector'],
                or same as 'domain' ('{}'), but is '{}'
                """.format(
                    domain, child.domain
                )
            )
        if name is None:
            name = "broadcast"
        super().__init__(name, child)
        # overwrite child domain ([]) with specified broadcasting domain
        self.domain = domain

        # set type of broadcast
        self.broadcast_type = broadcast_type

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

        return Broadcast(child, self.domain, broadcast_type=self.broadcast_type)

    def _unary_new_copy(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

        return Broadcast(child, self.domain, broadcast_type=self.broadcast_type)

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
        else:
            raise KeyError(
                """Broadcast type must be either: 'primary' or 'full' and not {}.
                 Support for 'secondary' will be added in the future""".format(
                    self.broadcast_ype
                )
            )
