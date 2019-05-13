#
# Concatenation classes
#
import pybamm

import numpy as np
from scipy.sparse import vstack
import copy


class Concatenation(pybamm.Symbol):
    """A node in the expression tree representing a concatenation of symbols

    **Extends**: :class:`pybamm.Symbol`

    Parameters
    ----------
    children : iterable of :class:`pybamm.Symbol`
        The symbols to concatenate

    """

    def __init__(self, *children, name=None, check_domain=True):
        if name is None:
            name = "concatenation"
        if check_domain:
            domain = self.get_children_domains(children)
        else:
            domain = []
        super().__init__(name, children, domain=domain)

    def get_children_domains(self, children):
        # combine domains from children
        domain = []
        for child in children:
            child_domain = child.domain
            if set(domain).isdisjoint(child_domain):
                domain += child_domain
            else:
                raise pybamm.DomainError("""domain of children must be disjoint""")
        return domain

    def _concatenation_evaluate(self, children_eval):
        """ Concatenate the evaluated children. """
        raise NotImplementedError

    def evaluate(self, t=None, y=None, known_evals=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        children = self.cached_children
        if known_evals is not None:
            if self.id not in known_evals:
                children_eval = [None] * len(children)
                for idx, child in enumerate(children):
                    children_eval[idx], known_evals = child.evaluate(t, y, known_evals)
                known_evals[self.id] = self._concatenation_evaluate(children_eval)
            return known_evals[self.id], known_evals
        else:
            children_eval = [None] * len(children)
            for idx, child in enumerate(children):
                children_eval[idx] = child.evaluate(t, y)
            return self._concatenation_evaluate(children_eval)

    def _concatenation_simplify(self, children):
        """ See :meth:`pybamm.Symbol.simplify()`. """
        new_symbol = self.__class__(*children)
        new_symbol.domain = []
        return new_symbol


class NumpyConcatenation(Concatenation):
    """A node in the expression tree representing a concatenation of equations, when we
    *don't* care about domains. The class :class:`pybamm.DomainConcatenation`, which
    *is* careful about domains and uses broadcasting where appropriate, should be used
    whenever possible instead.

    Upon evaluation, equations are concatenated using numpy concatenation.

    **Extends**: :class:`Concatenation`

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
                children[i] = child * pybamm.Vector(np.array([1]))
        super().__init__(*children, name="numpy concatenation", check_domain=False)

    def _concatenation_evaluate(self, children_eval):
        """ See :meth:`Concatenation._concatenation_evaluate()`. """
        if len(children_eval) == 0:
            return np.array([])
        else:
            return np.concatenate([child for child in children_eval])

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        children = self.cached_children
        if len(children) == 0:
            return pybamm.Scalar(0)
        else:
            return SparseStack(*[child.jac(variable) for child in children])


class DomainConcatenation(Concatenation):
    """A node in the expression tree representing a concatenation of symbols, being
    careful about domains.

    It is assumed that each child has a domain, and the final concatenated vector will
    respect the sizes and ordering of domains established in mesh keys

    **Extends**: :class:`pybamm.Concatenation`

    Parameters
    ----------

    children : iterable of :class:`pybamm.Symbol`
        The symbols to concatenate

    mesh : :class:`pybamm.BaseMesh`
        The underlying mesh for discretisation, used to obtain the number of mesh points
        in each domain.

    copy_this : :class:`pybamm.DomainConcatenation` (optional)
        if provided, this class is initialised by copying everything except the children
        from `copy_this`. `mesh` is not used in this case

    """

    def __init__(self, children, mesh, copy_this=None):
        # Convert any constant symbols in children to a Vector of the right size for
        # concatenation
        children = list(children)

        # Allow the base class to sort the domains into the correct order
        super().__init__(*children, name="domain concatenation")

        # ensure domain is sorted according to mesh keys
        domain_dict = {d: mesh.domain_order.index(d) for d in self.domain}
        self.domain = sorted(domain_dict, key=domain_dict.__getitem__)

        if copy_this is None:
            # store mesh
            self._mesh = mesh

            # Check that there is a domain, otherwise the functionality won't work
            # and we should raise a DomainError
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
            self._children_slices = [
                self.create_slices(child) for child in self.cached_children
            ]
        else:
            self._mesh = copy.copy(copy_this._mesh)
            self._slices = copy.copy(copy_this._slices)
            self._size = copy.copy(copy_this._size)
            self._children_slices = copy.copy(copy_this._children_slices)

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

    def _concatenation_evaluate(self, children_eval):
        """ See :meth:`Concatenation._concatenation_evaluate()`. """
        # preallocate vector
        vector = np.empty(self._size)

        # loop through domains of children writing subvectors to final vector
        for child_vector, slices in zip(children_eval, self._children_slices):
            for child_dom, child_slice in slices.items():
                vector[self._slices[child_dom]] = child_vector[child_slice]

        return vector

    def jac(self, variable):
        """ See :meth:`pybamm.Symbol.jac()`. """
        children = self.cached_children
        if len(children) == 0:
            return pybamm.Scalar(0)
        else:
            return SparseStack(*[child.jac(variable) for child in children])

    def _concatenation_simplify(self, children):
        """ See :meth:`pybamm.Symbol.simplify()`. """
        # Simplify Concatenation of StateVectors to a single StateVector
        if all([isinstance(x, pybamm.StateVector) for x in children]) and all(
            [
                children[idx].y_slice.stop == children[idx + 1].y_slice.start
                for idx in range(len(children) - 1)
            ]
        ):
            return pybamm.StateVector(
                slice(children[0].y_slice.start, children[-1].y_slice.stop)
            )

        new_symbol = self.__class__(children, self.mesh, self)

        # TODO: this should not be needed, but somehow we are still getting domains in
        # the simplified children
        new_symbol.domain = []

        return new_symbol


class SparseStack(Concatenation):
    """A node in the expression tree representing a concatenation of sparse
    matrices. As with NumpyConcatenation, we *don't* care about domains.
    The class :class:`pybamm.DomainConcatenation`, which *is* careful about
    domains and uses broadcasting where appropriate, should be used whenever
    possible instead.

    **Extends**: :class:`Concatenation`

    Parameters
    ----------
    children : iterable of :class:`Concatenation`
        The equations to concatenate

    """

    def __init__(self, *children):
        children = list(children)
        super().__init__(*children, name="sparse stack", check_domain=False)

    def evaluate(self, t=None, y=None, known_evals=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        children = self.cached_children
        if known_evals is not None:
            if self.id not in known_evals:
                children_eval = [None] * len(children)
                for idx, child in enumerate(children):
                    children_eval[idx], known_evals = child.evaluate(t, y, known_evals)
                known_evals[self.id] = self._concatenation_evaluate(children_eval)
            return known_evals[self.id], known_evals
        else:
            children_eval = [None] * len(children)
            for idx, child in enumerate(children):
                children_eval[idx] = child.evaluate(t, y)
            return self._concatenation_evaluate(children_eval)

    def _concatenation_evaluate(self, children_eval):
        """ See :meth:`Concatenation.evaluate()`. """
        return vstack(children_eval)
