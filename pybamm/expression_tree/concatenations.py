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

        for i, child in enumerate(children):
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
        Check that the node has the correct size, broadcasting to a node of the correct
        size if it has size 1 (according to its domain).

        Parameters
        ----------
        node: derived from :class:`Symbol`
            the sub-expression to process (node.is_constant() is true)

        """
        try:
            node_size = node.size
        except AttributeError:
            node_size = 0

        if node_size > 1:
            # Make sure node size is the same as the number of points specified for
            # broadcast. Note that npts_for_broadcast is set by the discretisation
            if node.shape[0] != sum(
                [mesh[dom].npts_for_broadcast for dom in node.domain]
            ):
                raise ValueError(
                    "Error: expression evaluated to a vector of incorrect length"
                )
            # Broadcast in space if the node had size 1
        else:
            node = pybamm.NumpyBroadcast(node, node.domain, mesh)
        return node

    def evaluate(self, t=None, y=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        try:
            return np.concatenate([child.evaluate(t, y) for child in self.children])
        except ValueError:
            import ipdb

            ipdb.set_trace()
        # preallocate vector
        vector = np.empty(self._size)

        # loop through domains of children writing subvectors to final vector
        for child, slices in zip(self.children, self._children_slices):
            child_vector = child.evaluate(t, y)
            for dom in child.domain:
                vector[self._slices[dom]] = child_vector[slices[dom]]

        return vector


class PiecewiseConstant(Concatenation):
    """Piecewise constant concatenation of three symbols.
    This is useful when we don't want to assign a domain to the inputs

    Parameters
    ----------
    neg_value: :class:`numbers.Number` or :class:`pybamm.Symbol`
        The value in the negative electrode
    sep_value: :class:`numbers.Number` or :class:`pybamm.Symbol`
        The value in the separator
    pos_value: :class:`numbers.Number` or :class:`pybamm.Symbol`
        The value in the positive electrode

    """

    def __init__(self, neg_value, sep_value, pos_value):
        neg_value_with_domain = pybamm.Broadcast(neg_value, ["negative electrode"])
        sep_value_with_domain = pybamm.Broadcast(sep_value, ["separator"])
        pos_value_with_domain = pybamm.Broadcast(pos_value, ["positive electrode"])
        super().__init__(
            neg_value_with_domain, sep_value_with_domain, pos_value_with_domain
        )
