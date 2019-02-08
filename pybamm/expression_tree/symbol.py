#
# Base Symbol Class for the expression tree
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import anytree
import numbers
import copy

from anytree.exporter import DotExporter


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
        self._name = name

        for child in children:
            # copy child before adding
            # this also adds copy.copy(child) to self.children
            copy.copy(child).parent = self

        self.domain = domain

    @property
    def name(self):
        """name of the node"""
        return self._name

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

    @property
    def id(self):
        """
        The immutable "identity" of a variable (for identifying y_slices).

        This is identical to what we'd put in a __hash__ function
        However, implementing __hash__ requires also implementing __eq__,
        which would then mess with loop-checking in the anytree module
        """
        return hash(
            (self.__class__, self.name)
            + tuple([child.id for child in self.children])
            + tuple(self.domain)
        )

    def render(self):
        """print out a visual representation of the tree (this node and its
        children)
        """
        for pre, _, node in anytree.RenderTree(self):
            print("%s%s" % (pre, str(node)))

    def visualise(self, filename, test=False):
        """Produces a .png file of the tree (this node and its children) with the
        name filename"""

        new_node, counter = self.relabel_tree(self, 0)

        # check that filename ends in .png.
        filename = "view_tree/" + filename + ".png"

        if test is False:
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

        if isinstance(symbol, pybamm.BinaryOperator):
            left, right = symbol.children
            new_left, counter = self.relabel_tree(left, counter)
            new_right, counter = self.relabel_tree(right, counter)
            new_node.children = [new_left, new_right]

        elif isinstance(symbol, pybamm.UnaryOperator):
            new_child, counter = self.relabel_tree(symbol.children[0], counter)
            new_node.children = [new_child]

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

    def evaluate(self, t=None, y=None):
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
        return all([not isinstance(n, search_types) for n in self.pre_order()])

    def evaluates_to_number(self):
        """Returns True if evaluating the expression returns a number.
        Returns False otherwise, including if NotImplementedError is raised.
        !Not to be confused with isinstance(self, pybamm.Scalar)!

        See Also
        --------
        evaluate : evaluate the expression

        """
        try:
            # return true if node evaluates to a number
            return isinstance(self.evaluate(), numbers.Number)
        except NotImplementedError:
            # return false if NotImplementedError is raised
            # (there is a e.g. Parameter, Variable, ... in the tree)
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
