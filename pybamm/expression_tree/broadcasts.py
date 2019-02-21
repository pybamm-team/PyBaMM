#
# Unary operator classes and methods
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm
import numpy as np


class TimeBroadcast(pybamm.UnaryOperator):
    def __init__(self, child):
        """ See :meth:`pybamm.UnaryOperator.__init__()`. """
        super().__init__("time broadcast", child)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        child_eval = child.evaluate(t, y)
        raise NotImplementedError


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
    name : string
        name of the node

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, domain, name=None):
        if child.domain not in [[], domain]:
            raise pybamm.DomainError(
                """
                Domain of a broadcasted child must be []
                or same as 'domain' but is '{}'
                """.format(
                    child.domain
                )
            )
        if name is None:
            name = "broadcast"
        super().__init__(name, child)
        # overwrite child domain ([]) with specified broadcasting domain
        self.domain = domain


class NumpyBroadcast(Broadcast):
    """A node in the expression tree implementing a broadcasting operator using numpy.
    Broadcasts a child (which *must* have empty domain) to a specified domain. To do
    this, creates a np array of ones of the same shape as the submesh domain, and then
    multiplies the child by that array upon evaluation

    Parameters
    ----------
    child : :class:`Symbol`
        child node
    domain : iterable of string
        the domain to broadcast the child to
    mesh : :class:`pybamm.Mesh`
        the mesh on which to broadcast

    **Extends:** :class:`SpatialOperator`
    """

    def __init__(self, child, domain, mesh):
        super().__init__(child, domain, name="numpy broadcast")
        # determine broadcasting vector size (size 1 if the domain is empty)
        if domain == []:
            self.broadcasting_vector_size = 1
        else:
            self.broadcasting_vector_size = sum(
                [mesh[dom].npts_for_broadcast for dom in domain]
            )
        # create broadcasting vector (vector of ones with shape determined by the
        # domain)
        self.broadcasting_vector = np.ones(self.broadcasting_vector_size)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        child = self.children[0]
        child_eval = child.evaluate(t, y)
        # if child is a vector, add a dimension for broadcasting
        if isinstance(child, pybamm.Vector):
            return child_eval[:, np.newaxis] * self.broadcasting_vector
        # if child is a state vector, check that it has the right shape and then
        # broadcast
        elif isinstance(child, pybamm.StateVector):
            assert child_eval.shape[0] == 1, ValueError(
                """child_eval should have shape (1,n), not {}""".format(
                    child_eval.shape
                )
            )
            return np.repeat(child_eval, self.broadcasting_vector_size, axis=0)
        # otherwise just do normal multiplication
        else:
            return child_eval * self.broadcasting_vector
