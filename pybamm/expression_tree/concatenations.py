#
# Concatenation classes
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import numpy as np
import numbers


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

    @property
    def children(self):
        children_list = []
        for child in tuple(self._NodeMixin__children_):
            if isinstance(child, pybamm.Variable):
                child = pybamm.Variable(child.name, child.domain)
            children_list.append(child)
        return children_list


class NumpyModelConcatenation(pybamm.Symbol):
    """A node in the expression tree representing a concatenation of equations.
    Upon evaluation, equations are concatenated using numpy concatenation.
    Unlike :class:`pybamm.Concatenation`, this doesn't check domains, as its only use
    is to concatenate model equations (e.g. rhs equations or initial conditions, in
    :class:`pybamm.BaseDiscretisation`), which might have common domains

    **Extends**: :class:`pybamm.Symbol`

    Parameters
    ----------
    children : iterable of :class:`pybamm.Symbol`
        The equations to concatenate

    """

    def __init__(self, *children):
        children = list(children)
        for i, child in enumerate(children):
            if isinstance(child, pybamm.Scalar):
                children[i] = pybamm.Vector(np.array([child.value]))
        super().__init__("model concatenation", children, domain=[])

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        if len(self.children) == 0:
            return np.array([])
        else:
            return np.concatenate([child.evaluate(t, y) for child in self.children])


class DomainConcatenation(Concatenation):
    """A node in the expression tree representing a concatenation of symbols.

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

        for i, child in enumerate(children):
            if child.is_constant():
                children[i] = self.process_node_for_concatenate(child, mesh)

        # Allow the base class to sort the domains into the correct order
        super().__init__(*children, name="domain concatenation")

        # create dict of domain => slice of final vector
        self._slices = self.create_slices(self, mesh)

        # store size of final vector
        self._size = self._slices[self.domain[-1]].stop

        # create disc of domain => slice for each child
        self._children_slices = []
        for child in self.children:
            self._children_slices.append(self.create_slices(child, mesh))

    def create_slices(self, node, mesh):
        slices = {}
        start = 0
        end = 0
        for dom in node.domain:
            end += mesh[dom].npts
            slices[dom] = slice(start, end)
            start = end
        return slices

    def process_node_for_concatenate(self, node, mesh):
        """
        the node is assumed to be constant in time. this function replaces it with a
        single Vector node with the correct length vector (according to its domain)

        Parameters
        ----------
        node: derived from :class:`Symbol`
            the sub-expression to process (node.is_constant() is true)

        """

        # node must be constant
        value = node.evaluate()

        # correct size of vector should be number of points in the domains
        subvector_size = sum([mesh[dom].npts for dom in node.domain])

        # check if its a scalar, if so convert to vector
        if isinstance(value, numbers.Number):
            value = np.full(subvector_size, value)

        # check it is the right size
        if value.size != subvector_size:
            raise ValueError(
                "Error: expression evaluated to a vector of incorrect length"
            )

        # convert to a Vector node
        return pybamm.Vector(value, domain=node.domain)

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
