#
# Unary operator classes and methods
#
import pybamm

import numbers
import numpy as np
from scipy.sparse import csr_matrix


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

    def __init__(self, child, domain, name=None):
        # Convert child to Scalar if it is a number
        if isinstance(child, numbers.Number):
            child = pybamm.Scalar(child)
        # Check domain
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

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

        return self.__class__(child, self.domain)


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
            vector_size = 0
            for dom in domain:
                # just create a vector of the points even in 2 and 3D
                for i in range(len(mesh[dom])):
                    vector_size += mesh[dom][i].npts_for_broadcast
            self.broadcasting_vector_size = vector_size
        # create broadcasting vector (vector of ones with shape determined by the
        # domain)
        self.broadcasting_vector = np.ones(self.broadcasting_vector_size)

        # store mesh
        self._mesh = mesh

    @property
    def mesh(self):
        return self._mesh

    def _unary_evaluate(self, child):
        """ See :meth:`pybamm.UnaryOperator._unary_evaluate()`. """
        return child * self.broadcasting_vector

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        child = self.orphans[0]
        if child.evaluates_to_number():
            variable_y_indices = np.arange(
                variable.y_slice.start, variable.y_slice.stop
            )
            jac = csr_matrix(
                (self.broadcasting_vector_size, np.size(variable_y_indices))
            )
            return pybamm.Matrix(jac)
        else:
            return child.jac(variable)

    def _unary_simplify(self, child):
        """ See :meth:`pybamm.UnaryOperator.simplify()`. """

        return self.__class__(child, self.domain, self.mesh)
