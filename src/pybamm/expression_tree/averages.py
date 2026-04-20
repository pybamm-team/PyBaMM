#
# Classes and methods for averaging
#
from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import pybamm


def _is_independent_of(
    symbol: pybamm.Symbol, domain_matches: Callable[[str], bool]
) -> bool:
    """True if no Variable/SpatialVariable leaf has a primary domain that
    ``domain_matches`` matches. Broadcasts from a non-matching domain are
    therefore treated as independent even if the broadcast itself carries a
    matching domain.
    """
    for node in symbol.pre_order():
        if isinstance(node, pybamm.Variable | pybamm.SpatialVariable) and any(
            domain_matches(dom) for dom in node.domain
        ):
            return False
    return True


class _BaseAverage(pybamm.Integral):
    """
    Base class for a symbol representing an average

    Parameters
    -----------
    child : :class:`pybamm.Symbol`
        The child node
    """

    def __init__(
        self,
        child: pybamm.Symbol,
        name: str,
        integration_variable: (
            list[pybamm.IndependentVariable] | pybamm.IndependentVariable
        ),
    ) -> None:
        super().__init__(child, integration_variable)
        self.name = name

    @classmethod
    def domain_matches(cls, d: str) -> bool:
        """Return True if domain string ``d`` is relevant for this average type."""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def symbol_is_constant(cls, symbol: pybamm.Symbol) -> bool:
        """Return True if ``symbol`` is independent of this average's domain."""
        return _is_independent_of(symbol, cls.domain_matches)

    @classmethod
    def _try_separable(
        cls,
        symbol: pybamm.Symbol,
        average_fn,
    ) -> pybamm.Symbol | None:
        """Rewrite ``avg(symbol)`` using linearity and constant-factor pull-out.

        * ``Addition`` / ``Subtraction`` always split: ``avg(a±b) = avg(a) ± avg(b)``.
        * ``Multiplication`` / ``Division`` split when at least one operand is
          constant under this average.

        Returns ``None`` when no rule applies.
        """
        operator = type(symbol)
        if isinstance(symbol, pybamm.Addition | pybamm.Subtraction):
            left, right = symbol.orphans
            return operator(average_fn(left), average_fn(right))
        if isinstance(symbol, pybamm.Multiplication | pybamm.Division):
            left, right = symbol.orphans
            if cls.symbol_is_constant(left) or cls.symbol_is_constant(right):
                return operator(average_fn(left), average_fn(right))
        return None

    @classmethod
    def from_symbol(cls, symbol: pybamm.Symbol) -> pybamm.Symbol:
        """Create average from symbol with simplifications."""
        raise NotImplementedError  # pragma: no cover


class XAverage(_BaseAverage):
    DOMAINS: ClassVar[tuple[str, ...]] = (
        "negative electrode",
        "separator",
        "positive electrode",
    )

    def __init__(self, child: pybamm.Symbol) -> None:
        if all(n in child.domain[0] for n in ["negative", "particle"]):
            x = pybamm.standard_spatial_vars.x_n
        elif all(n in child.domain[0] for n in ["positive", "particle"]):
            x = pybamm.standard_spatial_vars.x_p
        else:
            x = pybamm.SpatialVariable("x", domain=child.domain)
        super().__init__(child, "x-average", x)

    def _unary_new_copy(
        self, child: pybamm.Symbol, perform_simplifications: bool = True
    ):
        if perform_simplifications:
            return self.from_symbol(child)
        return XAverage(child)

    @classmethod
    def domain_matches(cls, d: str) -> bool:
        return d in cls.DOMAINS

    @classmethod
    def from_symbol(cls, symbol: pybamm.Symbol) -> pybamm.Symbol:
        """Create x-average with simplifications."""
        # Can't take average if symbol evaluates on edges (unless broadcasted)
        if symbol.evaluates_on_edges("primary") and not isinstance(
            symbol, pybamm.Broadcast
        ):
            raise ValueError(
                "Can't take the x-average of a symbol that evaluates on edges"
            )

        # If symbol doesn't have an electrode domain, return unchanged
        if not any(
            any(dom in cls.DOMAINS for dom in domain)
            for domain in symbol.domains.values()
        ):
            return symbol

        # If symbol is a broadcast, reduce by one dimension
        if isinstance(
            symbol,
            pybamm.PrimaryBroadcast | pybamm.SecondaryBroadcast | pybamm.FullBroadcast,
        ):
            if all(dom in cls.DOMAINS for dom in symbol.broadcast_domain):
                return symbol.reduce_one_dimension()
            elif isinstance(symbol, pybamm.PrimaryBroadcast):
                return pybamm.PrimaryBroadcast(
                    cls.from_symbol(symbol.orphans[0]), symbol.broadcast_domain
                )
            elif isinstance(symbol, pybamm.FullBroadcast) and all(
                dom in cls.DOMAINS for dom in symbol.secondary_domain
            ):
                domains = {
                    "primary": symbol.domains["primary"],
                    "secondary": symbol.domains["tertiary"],
                    "tertiary": symbol.domains["quaternary"],
                }
                return pybamm.FullBroadcast(
                    symbol.orphans[0], broadcast_domains=domains
                )
            elif isinstance(symbol, pybamm.FullBroadcast) and all(
                dom in cls.DOMAINS for dom in symbol.tertiary_domain
            ):
                domains = {
                    "primary": symbol.domains["primary"],
                    "secondary": symbol.domains["secondary"],
                    "tertiary": symbol.domains["quaternary"],
                }
                return pybamm.FullBroadcast(
                    symbol.orphans[0], broadcast_domains=domains
                )
            else:  # pragma: no cover
                raise NotImplementedError

        # Concatenation: thickness-weighted average of children
        if isinstance(symbol, pybamm.Concatenation) and not isinstance(
            symbol, pybamm.ConcatenationVariable
        ):
            geo = pybamm.geometric_parameters
            ls = {
                ("negative electrode",): geo.n.L,
                ("separator",): geo.s.L,
                ("positive electrode",): geo.p.L,
                ("separator", "positive electrode"): geo.s.L + geo.p.L,
            }
            out = sum(
                ls[tuple(orp.domain)] * cls.from_symbol(orp) for orp in symbol.orphans
            ) / sum(ls[tuple(orp.domain)] for orp in symbol.orphans)
            return out

        # Linearity + constant-factor pull-out
        simplified = cls._try_separable(symbol, cls.from_symbol)
        if simplified is not None:
            return simplified

        return cls(symbol)


class ZAverage(_BaseAverage):
    DOMAINS: ClassVar[tuple[str, ...]] = ("current collector",)

    def __init__(self, child: pybamm.Symbol) -> None:
        integration_variable: list[pybamm.IndependentVariable] = [
            pybamm.standard_spatial_vars.z
        ]
        super().__init__(child, "z-average", integration_variable)

    def _unary_new_copy(
        self, child: pybamm.Symbol, perform_simplifications: bool = True
    ):
        if perform_simplifications:
            return self.from_symbol(child)
        return ZAverage(child)

    @classmethod
    def domain_matches(cls, d: str) -> bool:
        return d in cls.DOMAINS

    @classmethod
    def from_symbol(cls, symbol: pybamm.Symbol) -> pybamm.Symbol:
        """Create z-average with simplifications."""
        if symbol.evaluates_on_edges("primary"):
            raise ValueError(
                "Can't take the z-average of a symbol that evaluates on edges"
            )

        if symbol.domain not in [[], ["current collector"]]:
            raise pybamm.DomainError(
                "z-average only implemented in the 'current collector' domain, "
                f"but symbol has domains {symbol.domain}"
            )

        if symbol.domain == []:
            return symbol

        if isinstance(symbol, pybamm.Broadcast):
            return symbol.reduce_one_dimension()

        simplified = cls._try_separable(symbol, cls.from_symbol)
        if simplified is not None:
            return simplified

        return cls(symbol)


class YZAverage(_BaseAverage):
    DOMAINS: ClassVar[tuple[str, ...]] = ("current collector",)

    def __init__(self, child: pybamm.Symbol) -> None:
        y = pybamm.standard_spatial_vars.y
        z = pybamm.standard_spatial_vars.z
        integration_variable: list[pybamm.IndependentVariable] = [y, z]
        super().__init__(child, "yz-average", integration_variable)

    def _unary_new_copy(
        self, child: pybamm.Symbol, perform_simplifications: bool = True
    ):
        if perform_simplifications:
            return self.from_symbol(child)
        return YZAverage(child)

    @classmethod
    def domain_matches(cls, d: str) -> bool:
        return d in cls.DOMAINS

    @classmethod
    def from_symbol(cls, symbol: pybamm.Symbol) -> pybamm.Symbol:
        """Create yz-average with simplifications."""
        if symbol.domain not in [[], ["current collector"]]:
            raise pybamm.DomainError(
                "y-z-average only implemented in the 'current collector' domain, "
                f"but symbol has domains {symbol.domain}"
            )

        if symbol.domain == []:
            return symbol

        if isinstance(symbol, pybamm.Broadcast):
            return symbol.reduce_one_dimension()

        simplified = cls._try_separable(symbol, cls.from_symbol)
        if simplified is not None:
            return simplified

        return cls(symbol)


class RAverage(_BaseAverage):
    def __init__(self, child: pybamm.Symbol) -> None:
        integration_variable: list[pybamm.IndependentVariable] = [
            pybamm.SpatialVariable("r", child.domain)
        ]
        super().__init__(child, "r-average", integration_variable)

    def _unary_new_copy(
        self, child: pybamm.Symbol, perform_simplifications: bool = True
    ):
        if perform_simplifications:
            return self.from_symbol(child)
        return RAverage(child)

    @classmethod
    def domain_matches(cls, d: str) -> bool:
        return d.endswith("particle") and not d.endswith("particle size")

    @classmethod
    def from_symbol(cls, symbol: pybamm.Symbol) -> pybamm.Symbol:
        """Create r-average with simplifications."""
        has_particle_domain = symbol.domain != [] and symbol.domain[0].endswith(
            "particle"
        )

        if symbol.evaluates_on_edges("primary"):
            raise ValueError(
                "Can't take the r-average of a symbol that evaluates on edges"
            )

        if not has_particle_domain:
            return symbol

        # SecondaryBroadcast onto electrode: r-average child then broadcast back
        if isinstance(symbol, pybamm.SecondaryBroadcast) and symbol.domains[
            "secondary"
        ] in [["positive electrode"], ["negative electrode"]]:
            child = symbol.orphans[0]
            child_av = cls.from_symbol(child)
            return pybamm.PrimaryBroadcast(child_av, symbol.domains["secondary"])

        # PrimaryBroadcast/FullBroadcast onto particle domain: reduce
        if (
            isinstance(symbol, pybamm.PrimaryBroadcast | pybamm.FullBroadcast)
            and has_particle_domain
        ):
            return symbol.reduce_one_dimension()

        simplified = cls._try_separable(symbol, cls.from_symbol)
        if simplified is not None:
            return simplified

        return cls(symbol)


class SizeAverage(_BaseAverage):
    """Size average uses weighted distribution. Does NOT support separable rewrite
    because the weight (f_a_dist) depends on the symbol's domain and cannot be
    meaningfully reassigned to sub-expressions.
    """

    DOMAINS: ClassVar[tuple[list[str], ...]] = (
        ["negative particle size"],
        ["positive particle size"],
        ["negative primary particle size"],
        ["positive primary particle size"],
        ["negative secondary particle size"],
        ["positive secondary particle size"],
    )

    def __init__(self, child: pybamm.Symbol, f_a_dist) -> None:
        R = pybamm.SpatialVariable("R", domains=child.domains, coord_sys="cartesian")
        integration_variable: list[pybamm.IndependentVariable] = [R]
        super().__init__(child, "size-average", integration_variable)
        self.f_a_dist = f_a_dist

    def _unary_new_copy(
        self, child: pybamm.Symbol, perform_simplifications: bool = True
    ):
        if perform_simplifications:
            return self.from_symbol(child, f_a_dist=self.f_a_dist)
        return SizeAverage(child, f_a_dist=self.f_a_dist)

    @classmethod
    def domain_matches(cls, d: str) -> bool:
        return d.endswith("particle size")

    @classmethod
    def _has_size_domain(cls, symbol: pybamm.Symbol) -> bool:
        """Check if symbol has any particle size domain."""
        return any(
            list(domain) in list(cls.DOMAINS) for domain in symbol.domains.values()
        )

    @classmethod
    def _get_f_a_dist(cls, symbol: pybamm.Symbol) -> pybamm.Symbol | None:
        """Compute area-weighted distribution for the symbol's domain."""
        geo = pybamm.geometric_parameters
        name = "R"
        if "negative" in symbol.domain[0]:
            name += "_n"
        elif "positive" in symbol.domain[0]:
            name += "_p"
        if "primary" in symbol.domain[0]:
            name += "_prim"
        elif "secondary" in symbol.domain[0]:
            name += "_sec"

        R = pybamm.SpatialVariable(name, domains=symbol.domains, coord_sys="cartesian")

        domains = symbol.domains
        if ["negative particle size"] in domains.values() or [
            "negative primary particle size"
        ] in domains.values():
            return geo.n.prim.f_a_dist(R)
        if ["negative secondary particle size"] in domains.values():
            return geo.n.sec.f_a_dist(R)
        if ["positive particle size"] in domains.values() or [
            "positive primary particle size"
        ] in domains.values():
            return geo.p.prim.f_a_dist(R)
        if ["positive secondary particle size"] in domains.values():
            return geo.p.sec.f_a_dist(R)
        return None  # pragma: no cover

    @classmethod
    def from_symbol(
        cls, symbol: pybamm.Symbol, f_a_dist: pybamm.Symbol | None = None
    ) -> pybamm.Symbol:
        """Create size-average with simplifications.

        Note: Does NOT use separable rewrite because the weighted average
        with distribution-dependent weight cannot be naively split without
        breaking conservation.
        """
        if symbol.evaluates_on_edges("primary"):
            raise ValueError(
                "Can't take the size-average of a symbol that evaluates on edges"
            )

        # If no size domain, return unchanged
        if symbol.domain == [] or not cls._has_size_domain(symbol):
            return symbol

        # PrimaryBroadcast to particle size: return orphan
        if isinstance(symbol, pybamm.PrimaryBroadcast) and symbol.domain in [
            ["negative particle size"],
            ["positive particle size"],
        ]:
            return symbol.orphans[0]

        # SecondaryBroadcast to particle size: return orphan
        if isinstance(symbol, pybamm.SecondaryBroadcast) and symbol.domains[
            "secondary"
        ] in [["negative particle size"], ["positive particle size"]]:
            return symbol.orphans[0]

        # Compute f_a_dist if not provided
        if f_a_dist is None:
            f_a_dist = cls._get_f_a_dist(symbol)

        return cls(symbol, f_a_dist)


# Convenience functions (thin wrappers)
def x_average(symbol: pybamm.Symbol) -> pybamm.Symbol:
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
    return XAverage.from_symbol(symbol)


def z_average(symbol: pybamm.Symbol) -> pybamm.Symbol:
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
    return ZAverage.from_symbol(symbol)


def yz_average(symbol: pybamm.Symbol) -> pybamm.Symbol:
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
    return YZAverage.from_symbol(symbol)


def xyz_average(symbol: pybamm.Symbol) -> pybamm.Symbol:
    return YZAverage.from_symbol(XAverage.from_symbol(symbol))


def r_average(symbol: pybamm.Symbol) -> pybamm.Symbol:
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
    return RAverage.from_symbol(symbol)


def size_average(
    symbol: pybamm.Symbol, f_a_dist: pybamm.Symbol | None = None
) -> pybamm.Symbol:
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
    return SizeAverage.from_symbol(symbol, f_a_dist)


def xyzs_average(symbol: pybamm.Symbol) -> pybamm.Symbol:
    return xyz_average(size_average(symbol))
