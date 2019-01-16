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

    def evaluate(self, t, y):
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

        # Simplify domain if concatenation spans the whole cell
        if domain == ["negative electrode", "separator", "positive electrode"]:
            domain = ["whole cell"]
        return domain


class NumpyConcatenation(Concatenation):
    """A node in the expression tree representing a concatenation of symbols.
    Upon evaluation, symbols are concatenated using numpy concatenation.

    **Extends**: :class:`pybamm.Concatenation`

    Parameters
    ----------
    children : iterable of :class:`pybamm.Symbol`
        The symbols to concatenate

    """

    def __init__(self, *children):
        # Convert any Scalar symbols in children to Vector for concatenation
        children = list(children)
        for i, child in enumerate(children):
            if isinstance(child, pybamm.Scalar):
                children[i] = pybamm.Vector(np.array([child.value]))

        super().__init__(*children, name="numpy concatenation")

    def evaluate(self, t, y):
        """ See :meth:`pybamm.Symbol.evaluate()`. """
        return np.concatenate([child.evaluate(t, y) for child in self.children])
