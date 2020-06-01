#
# Concatenation classes
#
import copy
import numpy as np
import pybamm
from scipy.sparse import vstack
from collections import defaultdict


class Concatenation(pybamm.Symbol):
    """A node in the expression tree representing a concatenation of symbols

    **Extends**: :class:`pybamm.Symbol`

    Parameters
    ----------
    children : iterable of :class:`pybamm.Symbol`
        The symbols to concatenate

    """

    def __init__(self, *children, name=None, check_domain=True, concat_fun=None):
        if name is None:
            name = "concatenation"
        if check_domain:
            domain = self.get_children_domains(children)
            auxiliary_domains = self.get_children_auxiliary_domains(children)
        else:
            domain = []
            auxiliary_domains = {}
        self.concatenation_function = concat_fun
        super().__init__(
            name, children, domain=domain, auxiliary_domains=auxiliary_domains
        )

    def get_children_domains(self, children):
        # combine domains from children
        domain = []
        for child in children:
            if not isinstance(child, pybamm.Symbol):
                raise TypeError("{} is not a pybamm symbol".format(child))
            child_domain = child.domain
            if set(domain).isdisjoint(child_domain):
                domain += child_domain
            else:
                raise pybamm.DomainError("""domain of children must be disjoint""")
        return domain

    def _concatenation_evaluate(self, children_eval):
        """ See :meth:`Concatenation._concatenation_evaluate()`. """
        if len(children_eval) == 0:
            return np.array([])
        else:
            return self.concatenation_function(children_eval)

    def evaluate(self, t=None, y=None, y_dot=None, inputs=None, known_evals=None):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        children = self.cached_children
        if known_evals is not None:
            if self.id not in known_evals:
                children_eval = [None] * len(children)
                for idx, child in enumerate(children):
                    children_eval[idx], known_evals = child.evaluate(
                        t, y, y_dot, inputs, known_evals
                    )
                known_evals[self.id] = self._concatenation_evaluate(children_eval)
            return known_evals[self.id], known_evals
        else:
            children_eval = [None] * len(children)
            for idx, child in enumerate(children):
                children_eval[idx] = child.evaluate(t, y, y_dot, inputs)
            return self._concatenation_evaluate(children_eval)

    def new_copy(self):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        new_children = [child.new_copy() for child in self.children]
        return self._concatenation_new_copy(new_children)

    def _concatenation_new_copy(self, children):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        new_symbol = self.__class__(*children)
        return new_symbol

    def _concatenation_jac(self, children_jacs):
        """ Calculate the jacobian of a concatenation """
        return NotImplementedError

    def _concatenation_simplify(self, children):
        """ See :meth:`pybamm.Symbol.simplify()`. """
        new_symbol = self.__class__(*children)
        new_symbol.clear_domains()
        return new_symbol

    def _evaluate_for_shape(self):
        """ See :meth:`pybamm.Symbol.evaluate_for_shape` """
        if len(self.children) == 0:
            return np.array([])
        else:
            # Default: use np.concatenate
            concatenation_function = self.concatenation_function or np.concatenate
            return concatenation_function(
                [child.evaluate_for_shape() for child in self.children]
            )


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
        super().__init__(
            *children,
            name="numpy concatenation",
            check_domain=False,
            concat_fun=np.concatenate
        )

    def _concatenation_jac(self, children_jacs):
        """ See :meth:`pybamm.Concatenation.concatenation_jac()`. """
        children = self.cached_children
        if len(children) == 0:
            return pybamm.Scalar(0)
        else:
            return SparseStack(*children_jacs)

    def _concatenation_simplify(self, children):
        """ See :meth:`pybamm.Symbol.simplify()`. """
        # Turn a concatenation of concatenations into a single concatenation
        new_children = []
        for child in children:
            # extract any children from numpy concatenation
            if isinstance(child, NumpyConcatenation):
                new_children.extend(child.orphans)
            else:
                new_children.append(child)
        new_symbol = NumpyConcatenation(*new_children)
        new_symbol.clear_domains()
        return new_symbol


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

    full_mesh : :class:`pybamm.BaseMesh`
        The underlying mesh for discretisation, used to obtain the number of mesh points
        in each domain.

    copy_this : :class:`pybamm.DomainConcatenation` (optional)
        if provided, this class is initialised by copying everything except the children
        from `copy_this`. `mesh` is not used in this case

    """

    def __init__(self, children, full_mesh, copy_this=None):
        # Convert any constant symbols in children to a Vector of the right size for
        # concatenation
        children = list(children)

        # Allow the base class to sort the domains into the correct order
        super().__init__(*children, name="domain concatenation")

        # ensure domain is sorted according to mesh keys
        domain_dict = {d: full_mesh.domain_order.index(d) for d in self.domain}
        self.domain = sorted(domain_dict, key=domain_dict.__getitem__)

        if copy_this is None:
            # store mesh
            self._full_mesh = full_mesh

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
            self.secondary_dimensions_npts = self._get_auxiliary_domain_repeats(
                self.domains
            )
            self._slices = self.create_slices(self)

            # store size of final vector
            self._size = self._slices[self.domain[-1]][-1].stop

            # create disc of domain => slice for each child
            self._children_slices = [
                self.create_slices(child) for child in self.cached_children
            ]
        else:
            self._full_mesh = copy.copy(copy_this._full_mesh)
            self._slices = copy.copy(copy_this._slices)
            self._size = copy.copy(copy_this._size)
            self._children_slices = copy.copy(copy_this._children_slices)
            self.secondary_dimensions_npts = copy_this.secondary_dimensions_npts

    def _get_auxiliary_domain_repeats(self, auxiliary_domains):
        """
        Helper method to read the 'auxiliary_domain' meshes
        """
        if "secondary" in auxiliary_domains:
            sec_mesh_npts = self.full_mesh.combine_submeshes(
                *auxiliary_domains["secondary"]
            ).npts
        else:
            sec_mesh_npts = 1
        if "tertiary" in auxiliary_domains:
            tert_mesh_npts = self.full_mesh.combine_submeshes(
                *auxiliary_domains["tertiary"]
            ).npts
        else:
            tert_mesh_npts = 1
        return sec_mesh_npts * tert_mesh_npts

    @property
    def full_mesh(self):
        return self._full_mesh

    def create_slices(self, node):
        slices = defaultdict(list)
        start = 0
        end = 0
        second_pts = self._get_auxiliary_domain_repeats(self.domains)
        if second_pts != self.secondary_dimensions_npts:
            raise ValueError(
                """Concatenation and children must have the same number of
                points in secondary dimensions"""
            )
        for i in range(second_pts):
            for dom in node.domain:
                end += self.full_mesh[dom].npts
                slices[dom].append(slice(start, end))
                start = end
        return slices

    def _concatenation_evaluate(self, children_eval):
        """ See :meth:`Concatenation._concatenation_evaluate()`. """
        # preallocate vector
        vector = np.empty((self._size, 1))

        # loop through domains of children writing subvectors to final vector
        for child_vector, slices in zip(children_eval, self._children_slices):
            for child_dom, child_slice in slices.items():
                for i, _slice in enumerate(child_slice):
                    vector[self._slices[child_dom][i]] = child_vector[_slice]

        return vector

    def _concatenation_jac(self, children_jacs):
        """ See :meth:`pybamm.Concatenation.concatenation_jac()`. """
        # note that this assumes that the children are in the right order and only have
        # one domain each
        jacs = []
        for i in range(self.secondary_dimensions_npts):
            for child_jac, slices in zip(children_jacs, self._children_slices):
                if len(slices) > 1:
                    raise NotImplementedError(
                        """jacobian only implemented for when each child has
                        a single domain"""
                    )
                child_slice = next(iter(slices.values()))
                jacs.append(child_jac[child_slice[i]])
        return SparseStack(*jacs)

    def _concatenation_new_copy(self, children):
        """ See :meth:`pybamm.Symbol.new_copy()`. """
        new_symbol = self.__class__(children, self.full_mesh, self)
        return new_symbol

    def _concatenation_simplify(self, children):
        """ See :meth:`pybamm.Symbol.simplify()`. """
        # Simplify Concatenation of StateVectors to a single StateVector
        # The sum of the evalation arrays of the StateVectors must be exactly 1
        if all([isinstance(child, pybamm.StateVector) for child in children]):
            longest_eval_array = len(children[-1]._evaluation_array)
            eval_arrays = {}
            for child in children:
                eval_arrays[child] = np.concatenate(
                    [
                        child.evaluation_array,
                        np.zeros(longest_eval_array - len(child.evaluation_array)),
                    ]
                )
            if all(sum(array for array in eval_arrays.values()) == 1):
                return pybamm.StateVector(
                    slice(children[0].y_slices[0].start, children[-1].y_slices[-1].stop)
                )

        new_symbol = self.__class__(children, self.full_mesh, self)

        # TODO: this should not be needed, but somehow we are still getting domains in
        # the simplified children
        new_symbol.clear_domains()

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
        super().__init__(
            *children, name="sparse stack", check_domain=False, concat_fun=vstack
        )
