#
# Base Symbol Class for the expression tree
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import anytree
import numbers
import copy
import autograd.numpy as np

from anytree.exporter import DotExporter
from scipy.sparse import issparse


def simplify_if_constant(new_node):
    """
    Utility function to simplify an expression tree if it evalutes to a constant
    scalar, vector or matrix
    """
    if new_node.is_constant():
        result = new_node.evaluate_ignoring_errors()
        if result is not None:
            if isinstance(result, numbers.Number):
                return pybamm.Scalar(result, domain=new_node.domain)
            elif isinstance(result, np.ndarray) or issparse(result):
                if result.ndim == 1:
                    return pybamm.Vector(result, domain=new_node.domain)
                else:
                    return pybamm.Matrix(result, domain=new_node.domain)
    return new_node


class Symbol(anytree.NodeMixin):
    """Base node class for the expression tree

    Parameters
    ----------

    name : str
        name for the node
    children : iterable :class:`Symbol`, optional
        children to attach to this node, default to an empty list
    domain : iterable of str, or str
        list of domains over which the node is valid (empty list indicates the symbol
        is valid over all domains)

    """

    def __init__(self, name, children=[], domain=[]):
        super(Symbol, self).__init__()
        self.name = name

        for child in children:
            # copy child before adding
            # this also adds copy.copy(child) to self.children
            copy.copy(child).parent = self

        # Set domain (and hence id)
        self.domain = domain

        # useful flags
        self._has_left_ghost_cell = False
        self._has_right_ghost_cell = False

    @property
    def name(self):
        """name of the node"""
        return self._name

    @name.setter
    def name(self, value):
        assert isinstance(value, str)
        self._name = value

    @property
    def domain(self):
        """list of applicable domains

        Returns
        -------
            iterable of str
        """
        return self._domain

    @domain.setter
    def domain(self, domain):
        if isinstance(domain, str):
            domain = [domain]
        try:
            iter(domain)
        except TypeError:
            raise TypeError("Domain: argument domain is not iterable")
        else:
            # check that domains are all known domains
            try:
                indicies = [pybamm.KNOWN_DOMAINS.index(d) for d in domain]
            except ValueError:
                raise ValueError(
                    """domain "{}" is not in known domains ({})""".format(
                        domain, str(pybamm.KNOWN_DOMAINS)
                    )
                )

            # check that domains are sorted correctly
            is_sorted = all(a <= b for a, b in zip(indicies, indicies[1:]))
            if not is_sorted:
                raise ValueError(
                    """
                    domain "{}" is not sorted according to known domains ({})
                    """.format(
                        domain, str(pybamm.KNOWN_DOMAINS)
                    )
                )

            self._domain = domain
            # Update id since domain has changed
            self.set_id()

    @property
    def id(self):
        return self._id

    def set_id(self):
        """
        Set the immutable "identity" of a variable (e.g. for identifying y_slices).

        This is identical to what we'd put in a __hash__ function
        However, implementing __hash__ requires also implementing __eq__,
        which would then mess with loop-checking in the anytree module.

        Hashing can be slow, so we set the id when we create the node, and hence only
        need to hash once.
        """
        self._id = hash(
            (self.__class__, self.name)
            + tuple([child.id for child in self.children])
            + tuple(self.domain)
        )

    @property
    def orphans(self):
        """
        Returning deepcopies of the children, with parents removed to avoid corrupting
        the expression tree internal data
        """
        orp = []
        for child in self.children:
            new_child = copy.deepcopy(child)
            new_child.parent = None
            orp.append(new_child)
        return tuple(orp)

    def render(self):
        """print out a visual representation of the tree (this node and its
        children)
        """
        for pre, _, node in anytree.RenderTree(self):
            print("%s%s" % (pre, str(node.name)))

    def visualise(self, filename):
        """
        Produces a .png file of the tree (this node and its children) with the
        name filename

        Parameters
        ----------

        filename : str
            filename to output, must end in ".png"

        """

        # check that filename ends in .png.
        if filename[-4:] != ".png":
            raise ValueError("filename should end in .png")

        new_node, counter = self.relabel_tree(self, 0)

        DotExporter(
            new_node, nodeattrfunc=lambda node: 'label="{}"'.format(node.label)
        ).to_picture(filename)

    def relabel_tree(self, symbol, counter):
        """ Finds all children of a symbol and assigns them a new id so that they can be
                visualised properly using the graphviz output
        """
        name = symbol.name
        if name == "div":
            name = "&nabla;&sdot;"
        elif name == "grad":
            name = "&nabla;"
        elif name == "/":
            name = "&divide;"
        elif name == "*":
            name = "&times;"
        elif name == "-":
            name = "&minus;"
        elif name == "+":
            name = "&#43;"
        elif name == "**":
            name = "^"
        elif name == "epsilon_s":
            name = "&#603;"

        new_node = anytree.Node(str(counter), label=name)
        counter += 1

        new_children = []
        for child in symbol.children:
            new_child, counter = self.relabel_tree(child, counter)
            new_children.append(new_child)
        new_node.children = new_children

        return new_node, counter

    def pre_order(self):
        """returns an iterable that steps through the tree in pre-order
        fashion

        Examples
        --------

        >>> import pybamm
        >>> a = pybamm.Symbol('a')
        >>> b = pybamm.Symbol('b')
        >>> for node in (a*b).pre_order():
        ...     print(node.name)
        *
        a
        b

        """
        return anytree.PreOrderIter(self)

    def __str__(self):
        """return a string representation of the node and its children"""
        return self._name

    def __repr__(self):
        """returns the string `__class__(id, name, children, domain)`"""
        return "{!s}({}, {!s}, children={!s}, domain={!s})".format(
            self.__class__.__name__,
            hex(self.id),
            self._name,
            [str(child) for child in self.children],
            [str(subdomain) for subdomain in self.domain],
        )

    def __add__(self, other):
        """return an :class:`Addition` object"""
        if isinstance(other, (Symbol, numbers.Number)):
            return pybamm.Addition(self, other)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        """return an :class:`Addition` object"""
        if isinstance(other, (Symbol, numbers.Number)):
            return pybamm.Addition(other, self)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        """return a :class:`Subtraction` object"""
        if isinstance(other, (Symbol, numbers.Number)):
            return pybamm.Subtraction(self, other)
        else:
            raise NotImplementedError

    def __rsub__(self, other):
        """return a :class:`Subtraction` object"""
        if isinstance(other, (Symbol, numbers.Number)):
            return pybamm.Subtraction(other, self)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        """return a :class:`Multiplication` object"""
        if isinstance(other, (Symbol, numbers.Number)):
            return pybamm.Multiplication(self, other)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        """return a :class:`Multiplication` object"""
        if isinstance(other, (Symbol, numbers.Number)):
            return pybamm.Multiplication(other, self)
        else:
            raise NotImplementedError

    def __matmul__(self, other):
        """return a :class:`MatrixMultiplication` object"""
        if isinstance(other, (Symbol, numbers.Number)):
            return pybamm.MatrixMultiplication(self, other)
        else:
            raise NotImplementedError

    def __rmatmul__(self, other):
        """return a :class:`MatrixMultiplication` object"""
        if isinstance(other, (Symbol, numbers.Number)):
            return pybamm.MatrixMultiplication(other, self)
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        """return a :class:`Division` object"""
        if isinstance(other, (Symbol, numbers.Number)):
            return pybamm.Division(self, other)
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        """return a :class:`Division` object"""
        if isinstance(other, (Symbol, numbers.Number)):
            return pybamm.Division(other, self)
        else:
            raise NotImplementedError

    def __pow__(self, other):
        """return a :class:`Power` object"""
        if isinstance(other, (Symbol, numbers.Number)):
            return pybamm.Power(self, other)
        else:
            raise NotImplementedError

    def __rpow__(self, other):
        """return a :class:`Power` object"""
        if isinstance(other, (Symbol, numbers.Number)):
            return pybamm.Power(other, self)
        else:
            raise NotImplementedError

    def __neg__(self):
        """return a :class:`Negate` object"""
        return pybamm.Negate(self)

    def __abs__(self):
        """return an :class:`AbsoluteValue` object"""
        return pybamm.AbsoluteValue(self)

    def diff(self, variable):
        """
        Differentiate a symbol with respect to a variable. Default behaviour is to
        return `1` if differentiating with respect to yourself and zero otherwise.
        Binary and Unary Operators override this.

        Parameters
        ----------
        variable : :class:`pybamm.Symbol`
            The variable with respect to which to differentiate

        """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            return pybamm.Scalar(0)

    def jac(self, variable):
        """
        Differentiate a symbol with respect to a (slice of) a State Vector.
        Default behaviour is to return `1` if differentiating with respect to
        yourself and zero otherwise. Binary and Unary Operators override this.

        Parameters
        ----------
        variable : :class:`pybamm.Symbol`
            The variable with respect to which to differentiate

        """
        if variable.id == self.id:
            return pybamm.Scalar(1)
        else:
            return pybamm.Scalar(0)

    def _base_evaluate(self, t=None, y=None):
        """evaluate expression tree

        will raise a ``NotImplementedError`` if this member function has not
        been defined for the node. For example, :class:`Scalar` returns its
        scalar value, but :class:`Variable` will raise ``NotImplementedError``

        Parameters
        ----------

        t : float or numeric type, optional
            time at which to evaluate (default None)

        y : numpy.array, optional
            array to evaluate when solving (default None)

        """
        raise NotImplementedError(
            """method self.evaluate() not implemented
               for symbol {!s} of type {}""".format(
                self, type(self)
            )
        )

    def evaluate(self, t=None, y=None, known_evals=None):
        """Evaluate expression tree (wrapper for dict of known values).
        If the dict 'known_evals' is provided, the dict is searched for self.id; if
        self.id is in the keys, return that value; otherwise, evaluate using
        :meth:`_base_evaluate()` and add that value to known_evals

        Parameters
        ----------
        t : float or numeric type, optional
            time at which to evaluate (default None)
        y : numpy.array, optional
            array to evaluate when solving (default None)
        known_evals : dict, optional
            dictionary containing known values (default None)

        Returns
        -------
        number or array
            the node evaluated at (t,y)
        known_evals (if known_evals input is not None) : dict
            the dictionary of known values
        """
        if known_evals is not None:
            if self.id not in known_evals:
                known_evals[self.id] = self._base_evaluate(t, y)
            return known_evals[self.id], known_evals
        else:
            return self._base_evaluate(t, y)

    def is_constant(self):
        """returns true if evaluating the expression is not dependent on `t` or `y`

        See Also
        --------
        evaluate : evaluate the expression

        """
        # if any of the nodes are instances of any of these types, then the whole
        # expression depends on either t or y
        search_types = (pybamm.Variable, pybamm.StateVector, pybamm.IndependentVariable)

        # do the search, return true if no relevent nodes are found
        return all([not (isinstance(n, search_types)) for n in self.pre_order()])

    def evaluate_ignoring_errors(self):
        """
        Evaluates the expression. If a node exists in the tree that cannot be evaluated
        as a scalar or vectr (e.g. Parameter, Variable, StateVector), then None is
        returned. Otherwise the result of the evaluation is given

        See Also
        --------
        evaluate : evaluate the expression

        """
        try:
            result = self.evaluate(t=0)
        except NotImplementedError:
            # return false if NotImplementedError is raised
            # (there is a e.g. Parameter, Variable, ... in the tree)
            return None
        except TypeError as error:
            # return false if specific TypeError is raised
            # (there is a e.g. StateVector in the tree)
            if error.args[0] == "StateVector cannot evaluate input 'y=None'":
                return None
            else:
                raise error

        return result

    def evaluates_to_number(self):
        """
        Returns True if evaluating the expression returns a number.
        Returns False otherwise, including if NotImplementedError or TyperError
        is raised.
        !Not to be confused with isinstance(self, pybamm.Scalar)!

        See Also
        --------
        evaluate : evaluate the expression

        """
        result = self.evaluate_ignoring_errors()

        if isinstance(result, numbers.Number):
            return True
        else:
            return False

    def has_spatial_derivatives(self):
        """Returns True if equation has spatial derivatives (grad or div)."""
        return self.has_gradient() or self.has_divergence()

    def has_gradient_and_not_divergence(self):
        """Returns True if equation has a Gradient term and not Divergence term."""
        return self.has_gradient() and not self.has_divergence()

    def has_gradient(self):
        """Returns True if equation has a Gradient term."""
        return any([isinstance(symbol, pybamm.Gradient) for symbol in self.pre_order()])

    def has_divergence(self):
        """Returns True if equation has a Divergence term."""
        return any(
            [isinstance(symbol, pybamm.Divergence) for symbol in self.pre_order()]
        )

    def simplify(self):
        """
        Simplify the expression tree.

        This function recurses down the tree, applying any simplifications defined in
        classes derived from pybamm.Symbol. E.g. any expression multiplied by a
        pybamm.Scalar(0) will be simplified to a pybamm.Scalar(0)
        """

        new_symbol = copy.deepcopy(self)
        # strip out domain info by default, so that conflicting domains are not an issue
        # during simplification. This should only be run after the model is discretised,
        # after which domains are no longer an issue
        new_symbol.domain = []
        new_symbol.parent = None
        return simplify_if_constant(new_symbol)

    @property
    def has_left_ghost_cell(self):
        return self._has_left_ghost_cell

    @has_left_ghost_cell.setter
    def has_left_ghost_cell(self, value):
        assert isinstance(value, bool)
        self._has_left_ghost_cell = value

    @property
    def has_right_ghost_cell(self):
        return self._has_right_ghost_cell

    @has_right_ghost_cell.setter
    def has_right_ghost_cell(self, value):
        assert isinstance(value, bool)
        self._has_right_ghost_cell = value

    @property
    def shape(self):
        """Shape of an object, found by evaluating it with appropriate t and y"""
        min_y_size = max(
            x.y_slice.stop
            for x in self.pre_order()
            if isinstance(x, pybamm.StateVector)
        )
        y = np.ones(min_y_size)
        eval = self.evaluate(t=0, y=y)
        return eval.shape
