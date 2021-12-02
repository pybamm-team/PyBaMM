#
# Classes and methods for averaging
#
import pybamm


class _BaseAverage(pybamm.SpatialOperator):
    """
    Base class for a symbol representing an average

    Parameters
    -----------
    child : :class:`pybamm.Symbol`
        The child node
    """

    def __init__(self, child, name):
        # average of a child takes the domain from auxiliary domain of the child
        if child.auxiliary_domains != {}:
            domain = child.auxiliary_domains["secondary"]
            if "tertiary" in child.auxiliary_domains:
                auxiliary_domains = {"secondary": child.auxiliary_domains["tertiary"]}
                if "quaternary" in child.auxiliary_domains:
                    auxiliary_domains["tertiary"] = child.auxiliary_domains[
                        "quaternary"
                    ]
            else:
                auxiliary_domains = {}
        # if child has no auxiliary domain, integral removes domain
        else:
            domain = []
            auxiliary_domains = {}
        super().__init__(
            name, child, domain=domain, auxiliary_domains=auxiliary_domains
        )

    def _evaluate_for_shape(self):
        """See :meth:`pybamm.Symbol.evaluate_for_shape_using_domain()`"""
        return pybamm.evaluate_for_shape_using_domain(
            self.domain, self.auxiliary_domains
        )

    def _evaluates_on_edges(self, dimension):
        """See :meth:`pybamm.Symbol._evaluates_on_edges()`."""
        return False


class XAverage(_BaseAverage):
    def __init__(self, child):
        super().__init__(child, "x-average")
        if child.domain in [
            ["negative particle"],
            ["negative particle size"],
        ]:
            self.integration_variable = pybamm.standard_spatial_vars.x_n
        elif child.domain in [
            ["positive particle"],
            ["positive particle size"],
        ]:
            self.integration_variable = pybamm.standard_spatial_vars.x_p
        else:
            self.integration_variable = pybamm.SpatialVariable("x", domain=child.domain)

    def _unary_new_copy(self, child):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return x_average(child)


class YZAverage(_BaseAverage):
    def __init__(self, child):
        super().__init__(child, "yz-average")
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z
        self.integration_variable = [y, z]

    def _unary_new_copy(self, child):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return yz_average(child)


class RAverage(_BaseAverage):
    def __init__(self, child):
        super().__init__(child, "r-average")
        self.integration_variable = pybamm.SpatialVariable("r", child.domain)

    def _unary_new_copy(self, child):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return r_average(child)


class SizeAverage(_BaseAverage):
    def __init__(self, child):
        super().__init__(child, "size-average")
        self.integration_variable = pybamm.SpatialVariable(
            "R",
            domain=symbol.domain,
            auxiliary_domains=symbol.auxiliary_domains,
            coord_sys="cartesian",
        )

    def _unary_new_copy(self, child):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return size_average(child)


def x_average(symbol):
    """
    convenience function for creating an average in the x-direction.

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The function to be averaged

    Returns
    -------
    :class:`Symbol`
        the new averaged symbol
    """
    # Can't take average if the symbol evaluates on edges
    if symbol.evaluates_on_edges("primary"):
        raise ValueError("Can't take the x-average of a symbol that evaluates on edges")
    # If symbol doesn't have a domain, its average value is itself
    if symbol.domain in [[], ["current collector"]]:
        new_symbol = symbol.new_copy()
        new_symbol.parent = None
        return new_symbol
    # If symbol is a primary or full broadcast, reduce by one dimension
    if isinstance(symbol, (pybamm.PrimaryBroadcast, pybamm.FullBroadcast)):
        return symbol.reduce_one_dimension()
    # If symbol is a concatenation of Broadcasts, its average value is its child
    elif isinstance(symbol, pybamm.Concatenation) and all(
        isinstance(child, pybamm.Broadcast) for child in symbol.children
    ):
        geo = pybamm.geometric_parameters
        l_n = geo.l_n
        l_s = geo.l_s
        l_p = geo.l_p
        if symbol.domain == ["negative electrode", "separator", "positive electrode"]:
            a, b, c = [orp.orphans[0] for orp in symbol.orphans]
            out = (l_n * a + l_s * b + l_p * c) / (l_n + l_s + l_p)
        elif symbol.domain == ["separator", "positive electrode"]:
            b, c = [orp.orphans[0] for orp in symbol.orphans]
            out = (l_s * b + l_p * c) / (l_s + l_p)
        # To respect domains we may need to broadcast the child back out
        child = symbol.children[0]
        # If symbol being returned doesn't have empty domain, return it
        if out.domain != []:
            return out
        # Otherwise we may need to broadcast it
        elif child.auxiliary_domains == {}:
            return out
        else:
            domain = child.auxiliary_domains["secondary"]
            if "tertiary" not in child.auxiliary_domains:
                return pybamm.PrimaryBroadcast(out, domain)
            else:
                auxiliary_domains = {"secondary": child.auxiliary_domains["tertiary"]}
                return pybamm.FullBroadcast(out, domain, auxiliary_domains)
    # Otherwise, use Integral to calculate average value
    else:
        return XAverage(symbol)


def z_average(symbol):
    """
    convenience function for creating an average in the z-direction.

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The function to be averaged

    Returns
    -------
    :class:`Symbol`
        the new averaged symbol
    """
    # Can't take average if the symbol evaluates on edges
    if symbol.evaluates_on_edges("primary"):
        raise ValueError("Can't take the z-average of a symbol that evaluates on edges")
    # Symbol must have domain [] or ["current collector"]
    if symbol.domain not in [[], ["current collector"]]:
        raise pybamm.DomainError(
            """z-average only implemented in the 'current collector' domain,
            but symbol has domains {}""".format(
                symbol.domain
            )
        )
    # If symbol doesn't have a domain, its average value is itself
    if symbol.domain == []:
        new_symbol = symbol.new_copy()
        new_symbol.parent = None
        return new_symbol
    # If symbol is a Broadcast, its average value is its child
    elif isinstance(symbol, pybamm.Broadcast):
        return symbol.orphans[0]
    # Otherwise, define a ZAverage
    else:
        return ZAverage(symbol)


def yz_average(symbol):
    """
    convenience function for creating an average in the y-z-direction.

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The function to be averaged

    Returns
    -------
    :class:`Symbol`
        the new averaged symbol
    """
    # Symbol must have domain [] or ["current collector"]
    if symbol.domain not in [[], ["current collector"]]:
        raise pybamm.DomainError(
            """y-z-average only implemented in the 'current collector' domain,
            but symbol has domains {}""".format(
                symbol.domain
            )
        )
    # If symbol doesn't have a domain, its average value is itself
    if symbol.domain == []:
        new_symbol = symbol.new_copy()
        new_symbol.parent = None
        return new_symbol
    # If symbol is a Broadcast, its average value is its child
    elif isinstance(symbol, pybamm.Broadcast):
        return symbol.orphans[0]
    # Otherwise, define a YZAverage
    else:
        return YZAverage(symbol)


def r_average(symbol):
    """
    convenience function for creating an average in the r-direction.

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The function to be averaged

    Returns
    -------
    :class:`Symbol`
        the new averaged symbol
    """
    # Can't take average if the symbol evaluates on edges
    if symbol.evaluates_on_edges("primary"):
        raise ValueError("Can't take the r-average of a symbol that evaluates on edges")
    # Otherwise, if symbol doesn't have a particle domain,
    # its r-averaged value is itself
    elif symbol.domain not in [
        ["positive particle"],
        ["negative particle"],
        ["working particle"],
    ]:
        new_symbol = symbol.new_copy()
        new_symbol.parent = None
        return new_symbol
    # If symbol is a secondary broadcast onto "negative electrode" or
    # "positive electrode", take the r-average of the child then broadcast back
    elif isinstance(symbol, pybamm.SecondaryBroadcast) and symbol.domains[
        "secondary"
    ] in [["positive electrode"], ["negative electrode"], ["working electrode"]]:
        child = symbol.orphans[0]
        child_av = pybamm.r_average(child)
        return pybamm.PrimaryBroadcast(child_av, symbol.domains["secondary"])
    # If symbol is a Broadcast onto a particle domain, its average value is its child
    elif isinstance(symbol, pybamm.PrimaryBroadcast) and symbol.domain in [
        ["positive particle"],
        ["negative particle"],
        ["working particle"],
    ]:
        return symbol.orphans[0]
    else:
        return RAverage(symbol)


def size_average(symbol):
    """convenience function for averaging over particle size R using the area-weighted
    particle-size distribution.

    Parameters
    ----------
    symbol : :class:`pybamm.Symbol`
        The function to be averaged
    Returns
    -------
    :class:`Symbol`
        the new averaged symbol
    """
    # Can't take average if the symbol evaluates on edges
    if symbol.evaluates_on_edges("primary"):
        raise ValueError(
            """Can't take the size-average of a symbol that evaluates on edges"""
        )

    # If symbol doesn't have a domain, or doesn't have "negative particle size"
    #  or "positive particle size" as a domain, it's average value is itself
    if symbol.domain == [] or not any(
        domain in [["negative particle size"], ["positive particle size"]]
        for domain in list(symbol.domains.values())
    ):
        new_symbol = symbol.new_copy()
        new_symbol.parent = None
        return new_symbol

    # If symbol is a primary broadcast to "particle size", take the orphan
    elif isinstance(symbol, pybamm.PrimaryBroadcast) and symbol.domain in [
        ["negative particle size"],
        ["positive particle size"],
    ]:
        return symbol.orphans[0]
    # If symbol is a secondary broadcast to "particle size" from "particle",
    # take the orphan
    elif isinstance(symbol, pybamm.SecondaryBroadcast) and symbol.domains[
        "secondary"
    ] in [["negative particle size"], ["positive particle size"]]:
        return symbol.orphans[0]
    # Otherwise, define a SizeAverage
    else:
        return SizeAverage(symbol)
