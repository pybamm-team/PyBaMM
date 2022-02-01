#
# Classes and methods for averaging
#
import pybamm


class _BaseAverage(pybamm.Integral):
    """
    Base class for a symbol representing an average

    Parameters
    -----------
    child : :class:`pybamm.Symbol`
        The child node
    """

    def __init__(self, child, name, integration_variable):
        super().__init__(child, integration_variable)
        self.name = name


class XAverage(_BaseAverage):
    def __init__(self, child):
        if child.domain in [
            ["negative particle"],
            ["negative particle size"],
        ]:
            x = pybamm.standard_spatial_vars.x_n
        elif child.domain in [
            ["positive particle"],
            ["positive particle size"],
        ]:
            x = pybamm.standard_spatial_vars.x_p
        else:
            x = pybamm.SpatialVariable("x", domain=child.domain)
        integration_variable = x
        super().__init__(child, "x-average", integration_variable)

    def _unary_new_copy(self, child):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return x_average(child)


class YZAverage(_BaseAverage):
    def __init__(self, child):
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z
        integration_variable = [y, z]
        super().__init__(child, "yz-average", integration_variable)

    def _unary_new_copy(self, child):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return yz_average(child)


class ZAverage(_BaseAverage):
    def __init__(self, child):
        integration_variable = [pybamm.standard_spatial_vars.z]
        super().__init__(child, "z-average", integration_variable)

    def _unary_new_copy(self, child):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return z_average(child)


class RAverage(_BaseAverage):
    def __init__(self, child):
        integration_variable = [pybamm.SpatialVariable("r", child.domain)]
        super().__init__(child, "r-average", integration_variable)

    def _unary_new_copy(self, child):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return r_average(child)


class SizeAverage(_BaseAverage):
    def __init__(self, child, f_a_dist):
        R = pybamm.SpatialVariable(
            "R",
            domain=child.domain,
            auxiliary_domains=child.auxiliary_domains,
            coord_sys="cartesian",
        )
        integration_variable = [R]
        super().__init__(child, "size-average", integration_variable)
        self.f_a_dist = f_a_dist

    def _unary_new_copy(self, child):
        """See :meth:`UnaryOperator._unary_new_copy()`."""
        return size_average(child, f_a_dist=self.f_a_dist)


def x_average(symbol):
    """
    Convenience function for creating an average in the x-direction.

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
    # If symbol doesn't have an electrode domain, its x-averaged value is itself
    if not any(
        any(
            dom in ["negative electrode", "separator", "positive electrode"]
            for dom in domain
        )
        for domain in symbol.domains.values()
    ):
        return symbol
    # If symbol is a broadcast, reduce by one dimension
    if isinstance(
        symbol,
        (pybamm.PrimaryBroadcast, pybamm.SecondaryBroadcast, pybamm.FullBroadcast),
    ):
        if all(
            dom in ["negative electrode", "separator", "positive electrode"]
            for dom in symbol.broadcast_domain
        ):
            return symbol.reduce_one_dimension()
        elif isinstance(symbol, pybamm.PrimaryBroadcast):
            return pybamm.PrimaryBroadcast(
                x_average(symbol.orphans[0]), symbol.broadcast_domain
            )
        elif isinstance(symbol, pybamm.FullBroadcast) and all(
            dom in ["negative electrode", "separator", "positive electrode"]
            for dom in symbol.secondary_domain
        ):
            aux = {}
            if "tertiary" in symbol.auxiliary_domains:
                aux["secondary"] = symbol.auxiliary_domains["tertiary"]
            return pybamm.FullBroadcast(symbol.orphans[0], symbol.broadcast_domain, aux)
        elif (
            isinstance(symbol, pybamm.FullBroadcast)
            and "tertiary" in symbol.auxiliary_domains
            and all(
                dom in ["negative electrode", "separator", "positive electrode"]
                for dom in symbol.tertiary_domain
            )
        ):
            aux = {"secondary": symbol.auxiliary_domains["secondary"]}
            if "quaternary" in symbol.auxiliary_domains:
                aux["tertiary"] = symbol.auxiliary_domains["quaternary"]
            return pybamm.FullBroadcast(symbol.orphans[0], symbol.broadcast_domain, aux)
        else:  # pragma: no cover
            # It should be impossible to get here
            raise NotImplementedError
    # If symbol is a concatenation of Broadcasts, its average value is the
    # thickness-weighted average of the symbols being broadcasted
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
    Convenience function for creating an average in the z-direction.

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
        return symbol
    # If symbol is a Broadcast, its average value is its child
    elif isinstance(symbol, pybamm.Broadcast):
        return symbol.orphans[0]
    # Otherwise, define a ZAverage
    else:
        return ZAverage(symbol)


def yz_average(symbol):
    """
    Convenience function for creating an average in the y-z-direction.

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
        return symbol
    # If symbol is a Broadcast, its average value is its child
    elif isinstance(symbol, pybamm.Broadcast):
        return symbol.orphans[0]
    # Otherwise, define a YZAverage
    else:
        return YZAverage(symbol)


def r_average(symbol):
    """
    Convenience function for creating an average in the r-direction.

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
        return symbol
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


def size_average(symbol, f_a_dist=None):
    """Convenience function for averaging over particle size R using the area-weighted
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
        return symbol

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
        if f_a_dist is None:
            geo = pybamm.geometric_parameters
            R = pybamm.SpatialVariable(
                "R",
                domain=symbol.domain,
                auxiliary_domains=symbol.auxiliary_domains,
                coord_sys="cartesian",
            )
            if ["negative particle size"] in symbol.domains.values():
                f_a_dist = geo.f_a_dist_n(R)
            elif ["positive particle size"] in symbol.domains.values():
                f_a_dist = geo.f_a_dist_p(R)
        return SizeAverage(symbol, f_a_dist)
