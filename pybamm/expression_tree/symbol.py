#
# Base Symbol Class for the expression tree
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import anytree
import numbers
import copy


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
            for d in domain:
                assert d in pybamm.KNOWN_DOMAINS, ValueError(
                    """domain "{}" is not in known domains ({})""".format(
                        d, str(pybamm.KNOWN_DOMAINS)
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
            (self.__class__, self.name) + tuple([child.id for child in self.children])
        )

    def render(self):
        """print out a visual representation of the tree (this node and its
        children)
        """
        for pre, _, node in anytree.RenderTree(self):
            print("%s%s" % (pre, str(node)))

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
        """returns the string `__class__(id, name, parent expression)`"""
        return "{!s}({}, {!s}, {!s})".format(
            self.__class__.__name__, hex(self.id), self._name, self.parent
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
