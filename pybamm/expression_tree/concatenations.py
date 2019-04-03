#
# Concatenation classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np


class Concatenation(pybamm.Symbol):
    """A node in the expression tree representing a concatenation of symbols

    **Extends**: :class:`pybamm.Symbol`

    Parameters
    ----------
    children : iterable of :class:`pybamm.Symbol`
        The symbols to concatenate

    """

    def __init__(self, *children, name=None):
        if name is None:
            name = "concatenation"
        domain = self.get_children_domains(children)
        super().__init__(name, children, domain=domain)

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        raise NotImplementedError

    def get_children_domains(self, children):
        # combine domains from children
        domain = []
        for child in children:
            child_domain = child.domain
            if set(domain).isdisjoint(child_domain):
                domain += child_domain
            else:
                raise pybamm.DomainError("""domain of children must be disjoint""")

        # ensure domain is sorted according to KNOWN_DOMAINS
        domain_dict = {d: pybamm.KNOWN_DOMAINS.index(d) for d in domain}
        domain = sorted(domain_dict, key=domain_dict.__getitem__)

        return domain


class NumpyConcatenation(pybamm.Symbol):
    """A node in the expression tree representing a concatenation of equations, when we
    *don't* care about domains. The class :class:`pybamm.DomainConcatenation`, which
    *is* careful about domains and uses broadcasting where appropriate, should be used
    whenever possible instead.

    Upon evaluation, equations are concatenated using numpy concatenation.

    **Extends**: :class:`pybamm.Symbol`

    Parameters
    ----------
    children : iterable of :class:`pybamm.Symbol`
        The equations to concatenate

    """

    def __init__(self, *children):
        children = list(children)
        # Turn objects that evaluate to scalars to objects that evaluate to vectors,
        # so that we can concatenate them
        for i, child in enumerate(children):
            if child.evaluates_to_number():
                children[i] = pybamm.NumpyBroadcast(child, [], None)
        super().__init__("model concatenation", children, domain=[])

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        if len(self.children) == 0:
            return np.array([])
        else:
            return np.concatenate([child.evaluate(t, y) for child in self.children])


class DomainConcatenation(Concatenation):
    """A node in the expression tree representing a concatenation of symbols, being
    careful about domains.

    It is assumed that each child has a domain, and the final concatenated vector will
    respect the sizes and ordering of domains established in pybamm.KNOWN_DOMAINS

    **Extends**: :class:`pybamm.Concatenation`

    Parameters
    ----------

    children : iterable of :class:`pybamm.Symbol`
        The symbols to concatenate

    mesh : :class:`pybamm.BaseMesh` (or subclass)
        The underlying mesh for discretisation, used to obtain the number of mesh points
        in each domain.

    """

    def __init__(self, children, mesh):
        # Convert any constant symbols in children to a Vector of the right size for
        # concatenation

        children = list(children)

        # Allow the base class to sort the domains into the correct order
        super().__init__(*children, name="domain concatenation")

        # store mesh
        self._mesh = mesh

        # Check that there is a domain, otherwise the functionality won't work and we
        # should raise a DomainError
        if self.domain == []:
            raise pybamm.DomainError(
                """
                domain cannot be empty for a DomainConcatenation.
                Perhaps the children should have been Broadcasted first?
                """
            )

        # create dict of domain => slice of final vector
        self._slices = self.create_slices(self)

        # store size of final vector
        self._size = self._slices[self.domain[-1]].stop

        # create disc of domain => slice for each child
        self._children_slices = [self.create_slices(child) for child in self.children]

    @property
    def mesh(self):
        return self._mesh

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self.size,)

    def create_slices(self, node):
        slices = {}
        start = 0
        end = 0
        for dom in node.domain:
            prim_pts = self.mesh[dom][0].npts
            second_pts = len(self.mesh[dom])
            end += prim_pts * second_pts
            slices[dom] = slice(start, end)
            start = end
        return slices

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        # preallocate vector
        vector = np.empty(self._size)

        # loop through domains of children writing subvectors to final vector
        for child, slices in zip(self.children, self._children_slices):
            child_vector = child.evaluate(t, y)
            for dom in child.domain:
                vector[self._slices[dom]] = child_vector[slices[dom]]

        return vector
