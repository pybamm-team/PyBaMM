#
# Tests for the Unary Operator classes
#
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import pybamm
from pybamm.expression_tree.averages import (
    RAverage,
    SizeAverage,
    XAverage,
    YZAverage,
    ZAverage,
)
from tests import assert_domain_equal


class TestUnaryOperators:
    @given(st.integers())
    def test_x_average(self, random_integer):
        a = pybamm.Scalar(random_integer)
        average_a = pybamm.x_average(a)
        assert average_a == a

        # average of a broadcast is the child
        average_broad_a = pybamm.x_average(
            pybamm.PrimaryBroadcast(a, ["negative electrode"])
        )
        assert average_broad_a == pybamm.Scalar(random_integer)

        # average of a number times a broadcast is the number times the child
        average_two_broad_a = pybamm.x_average(
            2 * pybamm.PrimaryBroadcast(a, ["negative electrode"])
        )
        assert average_two_broad_a == pybamm.Scalar(2 * random_integer)
        average_t_broad_a = pybamm.x_average(
            pybamm.t * pybamm.PrimaryBroadcast(a, ["negative electrode"])
        )
        assert average_t_broad_a == (pybamm.t * random_integer)

        # full broadcasts
        average_broad_a = pybamm.x_average(
            pybamm.FullBroadcast(
                a,
                ["negative particle"],
                {
                    "secondary": "negative particle size",
                    "tertiary": "negative electrode",
                    "quaternary": "current collector",
                },
            )
        )
        average_broad_a_simp = pybamm.FullBroadcast(
            a,
            ["negative particle"],
            {"secondary": "negative particle size", "tertiary": "current collector"},
        )
        assert average_broad_a == average_broad_a_simp

        # x-average of concatenation of broadcasts
        conc_broad = pybamm.concatenation(
            pybamm.PrimaryBroadcast(1, ["negative electrode"]),
            pybamm.PrimaryBroadcast(2, ["separator"]),
            pybamm.PrimaryBroadcast(3, ["positive electrode"]),
        )
        average_conc_broad = pybamm.x_average(conc_broad)
        assert isinstance(average_conc_broad, pybamm.Division)
        assert average_conc_broad.domain == []
        # separator and positive electrode only (half-cell model)
        conc_broad = pybamm.concatenation(
            pybamm.PrimaryBroadcast(2, ["separator"]),
            pybamm.PrimaryBroadcast(3, ["positive electrode"]),
        )
        average_conc_broad = pybamm.x_average(conc_broad)
        assert isinstance(average_conc_broad, pybamm.Division)
        assert average_conc_broad.domain == []
        # with auxiliary domains
        conc_broad = pybamm.concatenation(
            pybamm.FullBroadcast(
                1,
                ["negative electrode"],
                auxiliary_domains={"secondary": "current collector"},
            ),
            pybamm.FullBroadcast(
                2, ["separator"], auxiliary_domains={"secondary": "current collector"}
            ),
            pybamm.FullBroadcast(
                3,
                ["positive electrode"],
                auxiliary_domains={"secondary": "current collector"},
            ),
        )
        average_conc_broad = pybamm.x_average(conc_broad)
        assert isinstance(average_conc_broad, pybamm.PrimaryBroadcast)
        assert average_conc_broad.domain == ["current collector"]
        conc_broad = pybamm.concatenation(
            pybamm.FullBroadcast(
                1,
                ["negative electrode"],
                auxiliary_domains={
                    "secondary": "current collector",
                    "tertiary": "test",
                },
            ),
            pybamm.FullBroadcast(
                2,
                ["separator"],
                auxiliary_domains={
                    "secondary": "current collector",
                    "tertiary": "test",
                },
            ),
            pybamm.FullBroadcast(
                3,
                ["positive electrode"],
                auxiliary_domains={
                    "secondary": "current collector",
                    "tertiary": "test",
                },
            ),
        )
        average_conc_broad = pybamm.x_average(conc_broad)
        assert isinstance(average_conc_broad, pybamm.FullBroadcast)
        assert_domain_equal(
            average_conc_broad.domains,
            {"primary": ["current collector"], "secondary": ["test"]},
        )

        # x-average of broadcast
        for domain in [["negative electrode"], ["separator"], ["positive electrode"]]:
            a = pybamm.Variable("a", domain=domain)
            x = pybamm.SpatialVariable("x", domain)
            av_a = pybamm.x_average(a)
            assert isinstance(av_a, pybamm.XAverage)
            assert av_a.integration_variable[0].domain == x.domain
            assert av_a.domain == []

        # whole electrode domain
        domain = ["negative electrode", "separator", "positive electrode"]
        a = pybamm.Variable("a", domain=domain)
        x = pybamm.SpatialVariable("x", domain)
        av_a = pybamm.x_average(a)
        assert isinstance(av_a, pybamm.XAverage)
        assert av_a.integration_variable[0].domain == x.domain
        assert av_a.domain == []

        a = pybamm.Variable("a", domain="new domain")
        av_a = pybamm.x_average(a)
        assert av_a == a

        # x-average of symbol that evaluates on edges raises error
        symbol_on_edges = pybamm.SpatialVariableEdge(
            "x_n", domain=["negative electrode"]
        )
        with pytest.raises(
            ValueError,
            match=r"Can't take the x-average of a symbol that evaluates on edges",
        ):
            pybamm.x_average(symbol_on_edges)

        # Particle domains
        a = pybamm.Symbol(
            "a",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        av_a = pybamm.x_average(a)
        assert a.domain == ["negative particle"]
        assert isinstance(av_a, pybamm.XAverage)

        a = pybamm.Symbol(
            "a",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )
        av_a = pybamm.x_average(a)
        assert a.domain == ["positive particle"]
        assert isinstance(av_a, pybamm.XAverage)

        # Addition or Subtraction
        a = pybamm.Variable("a", domain="domain")
        b = pybamm.Variable("b", domain="domain")
        assert pybamm.x_average(a + b) == pybamm.x_average(a) + pybamm.x_average(b)
        assert pybamm.x_average(a - b) == pybamm.x_average(a) - pybamm.x_average(b)

    def test_x_average_factors_out_x_constant_multiplications(self):
        f = pybamm.Variable("f", domain="negative electrode")

        assert pybamm.x_average(2 * f) == 2 * pybamm.XAverage(f)
        assert pybamm.x_average(f * 2) == pybamm.XAverage(f) * 2
        assert pybamm.x_average(2 / f) == 2 / pybamm.XAverage(f)
        assert pybamm.x_average(f / 2) == pybamm.XAverage(f) / 2

        assert pybamm.x_average(pybamm.t * f) == pybamm.t * pybamm.XAverage(f)
        assert pybamm.x_average(f / pybamm.t) == pybamm.XAverage(f) / pybamm.t

        g = pybamm.Variable("g", domain="current collector")
        broad_g = pybamm.PrimaryBroadcast(g, "negative electrode")
        assert XAverage.symbol_is_constant(broad_g)

        out = pybamm.x_average(broad_g * f)
        assert out == g * pybamm.XAverage(f)
        assert out.domain == ["current collector"]

        out_div = pybamm.x_average(broad_g / f)
        assert out_div == g / pybamm.XAverage(f)
        assert out_div.domain == ["current collector"]

    def test_x_average_does_not_split_when_both_sides_x_dependent(self):
        f1 = pybamm.Variable("f1", domain="negative electrode")
        f2 = pybamm.Variable("f2", domain="negative electrode")

        prod = pybamm.x_average(f1 * f2)
        assert isinstance(prod, pybamm.XAverage)
        xavg_nodes = [n for n in prod.pre_order() if isinstance(n, pybamm.XAverage)]
        assert len(xavg_nodes) == 1

        quot = pybamm.x_average(f1 / f2)
        assert isinstance(quot, pybamm.XAverage)
        xavg_nodes = [n for n in quot.pre_order() if isinstance(n, pybamm.XAverage)]
        assert len(xavg_nodes) == 1

    def test_x_average_factors_out_compound_x_constant(self):
        f = pybamm.Variable("f", domain="negative electrode")
        out = pybamm.x_average(2 * pybamm.t * f)
        assert out == 2 * pybamm.t * pybamm.XAverage(f)
        xavg_nodes = [n for n in out.pre_order() if isinstance(n, pybamm.XAverage)]
        assert len(xavg_nodes) == 1
        assert xavg_nodes[0].orphans[0] == f

    def test_is_x_constant_helper(self):
        assert XAverage.symbol_is_constant(pybamm.Scalar(2.0))
        assert XAverage.symbol_is_constant(pybamm.t)

        v_part = pybamm.Variable("v_part", domain="negative particle")
        r = pybamm.SpatialVariable("r", domain="negative particle")
        v_cc = pybamm.Variable("v_cc", domain="current collector")
        assert XAverage.symbol_is_constant(v_part)
        assert XAverage.symbol_is_constant(r)
        assert XAverage.symbol_is_constant(v_cc)

        v_n = pybamm.Variable("v_n", domain="negative electrode")
        v_s = pybamm.Variable("v_s", domain="separator")
        v_p = pybamm.Variable("v_p", domain="positive electrode")
        assert not XAverage.symbol_is_constant(v_n)
        assert not XAverage.symbol_is_constant(v_s)
        assert not XAverage.symbol_is_constant(v_p)
        assert not XAverage.symbol_is_constant(pybamm.standard_spatial_vars.x_n)

        broadcast = pybamm.PrimaryBroadcast(v_cc, "negative electrode")
        assert broadcast.domain == ["negative electrode"]
        assert XAverage.symbol_is_constant(broadcast)

        assert XAverage.symbol_is_constant(2 * pybamm.t * v_part + pybamm.Scalar(3))
        assert not XAverage.symbol_is_constant(2 * pybamm.t + v_n)
        assert not XAverage.symbol_is_constant(broadcast * v_n)

    @given(st.integers(min_value=-(2**63), max_value=2**63 - 1))
    def test_size_average(self, random_integer):
        # no domain
        a = pybamm.Scalar(random_integer)
        average_a = pybamm.size_average(a)
        assert average_a == a

        b = pybamm.FullBroadcast(
            1,
            ["negative particle"],
            {"secondary": "negative electrode", "tertiary": "current collector"},
        )
        # no "particle size" domain
        average_b = pybamm.size_average(b)
        assert average_b == b

        # primary or secondary broadcast to "particle size" domain
        average_a = pybamm.size_average(
            pybamm.PrimaryBroadcast(a, "negative particle size")
        )
        assert average_a.evaluate() == np.array([random_integer])

        a = pybamm.Symbol("a", domain="negative particle")
        average_a = pybamm.size_average(
            pybamm.SecondaryBroadcast(a, "negative particle size")
        )
        assert average_a == a

        for domain in [
            ["negative particle size"],
            ["positive particle size"],
            ["negative primary particle size"],
            ["positive primary particle size"],
            ["negative secondary particle size"],
            ["positive secondary particle size"],
        ]:
            a = pybamm.Symbol("a", domain=domain)
            R = pybamm.SpatialVariable("R", domain)
            av_a = pybamm.size_average(a)
            assert isinstance(av_a, pybamm.SizeAverage)
            assert av_a.integration_variable[0].domain == R.domain
            # domain list should now be empty
            assert av_a.domain == []

        # R-average of symbol that evaluates on edges raises error
        symbol_on_edges = pybamm.PrimaryBroadcastToEdges(1, "domain")
        with pytest.raises(
            ValueError,
            match=r"""Can't take the size-average of a symbol that evaluates on edges""",
        ):
            pybamm.size_average(symbol_on_edges)

    def test_r_average(self):
        a = pybamm.Scalar(1)
        average_a = pybamm.r_average(a)
        assert average_a == a

        average_broad_a = pybamm.r_average(
            pybamm.PrimaryBroadcast(a, ["negative particle"])
        )
        assert average_broad_a.evaluate() == np.array([1])

        for domain in [["negative particle"], ["positive particle"]]:
            a = pybamm.Symbol("a", domain=domain)
            r = pybamm.SpatialVariable("r", domain)
            av_a = pybamm.r_average(a)
            assert isinstance(av_a, pybamm.RAverage)
            assert av_a.integration_variable[0].domain == r.domain
            # electrode domains go to current collector when averaged
            assert av_a.domain == []

        # r-average of a symbol that is broadcast to x
        # takes the average of the child then broadcasts it
        a = pybamm.PrimaryBroadcast(1, "positive particle")
        broad_a = pybamm.SecondaryBroadcast(a, "positive electrode")
        average_broad_a = pybamm.r_average(broad_a)
        assert isinstance(average_broad_a, pybamm.PrimaryBroadcast)
        assert average_broad_a.domain == ["positive electrode"]
        assert average_broad_a.children[0] == pybamm.r_average(a)

        # r-average of symbol that evaluates on edges raises error
        symbol_on_edges = pybamm.PrimaryBroadcastToEdges(1, "domain")
        with pytest.raises(
            ValueError,
            match=r"Can't take the r-average of a symbol that evaluates on edges",
        ):
            pybamm.r_average(symbol_on_edges)

        # Addition or Subtraction
        a = pybamm.Variable("a", domain="domain")
        b = pybamm.Variable("b", domain="domain")
        assert pybamm.r_average(a + b) == pybamm.r_average(a) + pybamm.r_average(b)
        assert pybamm.r_average(a - b) == pybamm.r_average(a) - pybamm.r_average(b)

    @given(st.integers(min_value=-(2**63), max_value=2**63 - 1))
    def test_yz_average(self, random_integer):
        a = pybamm.Scalar(random_integer)
        z_average_a = pybamm.z_average(a)
        yz_average_a = pybamm.yz_average(a)
        assert z_average_a == a
        assert yz_average_a == a

        z_average_broad_a = pybamm.z_average(
            pybamm.PrimaryBroadcast(a, ["current collector"])
        )
        yz_average_broad_a = pybamm.yz_average(
            pybamm.PrimaryBroadcast(a, ["current collector"])
        )
        assert z_average_broad_a.evaluate() == np.array([random_integer])
        assert yz_average_broad_a.evaluate() == np.array([random_integer])

        a = pybamm.Variable("a", domain=["current collector"])
        y = pybamm.SpatialVariable("y", ["current collector"])
        z = pybamm.SpatialVariable("z", ["current collector"])

        yz_av_a = pybamm.yz_average(a)
        assert isinstance(yz_av_a, pybamm.YZAverage)
        assert yz_av_a.integration_variable[0].domain == y.domain
        assert yz_av_a.integration_variable[1].domain == z.domain
        assert yz_av_a.domain == []

        z_av_a = pybamm.z_average(a)
        assert isinstance(z_av_a, pybamm.ZAverage)
        assert z_av_a.integration_variable[0].domain == a.domain
        assert z_av_a.domain == []

        a = pybamm.Symbol("a", domain="bad domain")
        with pytest.raises(pybamm.DomainError):
            pybamm.z_average(a)
        with pytest.raises(pybamm.DomainError):
            pybamm.yz_average(a)

        # average of symbol that evaluates on edges raises error
        symbol_on_edges = pybamm.PrimaryBroadcastToEdges(1, "domain")
        with pytest.raises(
            ValueError,
            match=r"Can't take the z-average of a symbol that evaluates on edges",
        ):
            pybamm.z_average(symbol_on_edges)

        # Addition or Subtraction
        a = pybamm.Variable("a", domain="current collector")
        b = pybamm.Variable("b", domain="current collector")
        assert pybamm.yz_average(a + b) == pybamm.yz_average(a) + pybamm.yz_average(b)
        assert pybamm.yz_average(a - b) == pybamm.yz_average(a) - pybamm.yz_average(b)
        assert pybamm.z_average(a + b) == pybamm.z_average(a) + pybamm.z_average(b)
        assert pybamm.z_average(a - b) == pybamm.z_average(a) - pybamm.z_average(b)

    # --- Generalised constant-factor pull-out -------------------------------
    #
    # ``_separable_average`` pulls factors that are constant under the
    # integration out of every averaging operator (x, yz, z, r, size). The
    # intent is that codegen evaluates the constant side once as a scalar
    # rather than re-evaluating it at every integration node.

    def test_is_constant_predicates(self):
        # cc / r predicates pick up the right leaf domains.
        v_cc = pybamm.Variable("v_cc", domain="current collector")
        v_rn = pybamm.Variable("v_rn", domain="negative particle")
        v_rp = pybamm.Variable("v_rp", domain="positive particle")
        v_R = pybamm.Variable("v_R", domain="negative particle size")
        v_n = pybamm.Variable("v_n", domain="negative electrode")
        scalar = pybamm.Scalar(2.5)

        # cc-constant (ZAverage.symbol_is_constant uses same domain as YZAverage)
        assert ZAverage.symbol_is_constant(scalar)
        assert ZAverage.symbol_is_constant(v_n)
        assert ZAverage.symbol_is_constant(v_rn)
        assert ZAverage.symbol_is_constant(v_R)
        assert not ZAverage.symbol_is_constant(v_cc)
        assert not ZAverage.symbol_is_constant(pybamm.standard_spatial_vars.z)

        # r-constant: no "particle" leaf (but "particle size" is NOT "particle")
        assert RAverage.symbol_is_constant(scalar)
        assert RAverage.symbol_is_constant(v_cc)
        assert RAverage.symbol_is_constant(v_n)
        assert RAverage.symbol_is_constant(v_R)
        assert not RAverage.symbol_is_constant(v_rn)
        assert not RAverage.symbol_is_constant(v_rp)
        assert not RAverage.symbol_is_constant(
            pybamm.SpatialVariable("r", "negative particle")
        )

    def test_z_average_factors_out_cc_constant(self):
        v_cc = pybamm.Variable("v_cc", domain="current collector")

        # scalar factor
        assert pybamm.z_average(2 * v_cc) == 2 * pybamm.ZAverage(v_cc)
        assert pybamm.z_average(v_cc * 2) == pybamm.ZAverage(v_cc) * 2
        assert pybamm.z_average(v_cc / 2) == pybamm.ZAverage(v_cc) / 2
        assert pybamm.z_average(2 / v_cc) == 2 / pybamm.ZAverage(v_cc)

        # time factor
        assert pybamm.z_average(pybamm.t * v_cc) == pybamm.t * pybamm.ZAverage(v_cc)

        # two cc-dependent factors: NO split
        v2 = pybamm.Variable("v2", domain="current collector")
        out = pybamm.z_average(v_cc * v2)
        assert isinstance(out, pybamm.ZAverage)

    def test_yz_average_factors_out_cc_constant(self):
        v_cc = pybamm.Variable("v_cc", domain="current collector")

        assert pybamm.yz_average(2 * v_cc) == 2 * pybamm.YZAverage(v_cc)
        assert pybamm.yz_average(v_cc / 2) == pybamm.YZAverage(v_cc) / 2
        assert pybamm.yz_average(pybamm.t * v_cc) == pybamm.t * pybamm.YZAverage(v_cc)

        v2 = pybamm.Variable("v2", domain="current collector")
        assert isinstance(pybamm.yz_average(v_cc * v2), pybamm.YZAverage)

    def test_r_average_factors_out_r_constant(self):
        v_part = pybamm.Variable("v_part", domain="negative particle")

        # scalar and time are r-constant
        assert pybamm.r_average(2 * v_part) == 2 * pybamm.RAverage(v_part)
        assert pybamm.r_average(v_part * 2) == pybamm.RAverage(v_part) * 2
        assert pybamm.r_average(v_part / 2) == pybamm.RAverage(v_part) / 2
        assert pybamm.r_average(pybamm.t * v_part) == pybamm.t * pybamm.RAverage(v_part)

        # cc-domain variable broadcast onto the particle domain is r-constant.
        # The broadcast is also r-averaged (reducing the particle dimension),
        # so the pulled-out factor collapses back to ``v_cc``.
        v_cc = pybamm.Variable("v_cc", domain="current collector")
        broad_cc = pybamm.PrimaryBroadcast(v_cc, "negative particle")
        out = pybamm.r_average(broad_cc * v_part)
        assert out == v_cc * pybamm.RAverage(v_part)
        assert out.domain == ["current collector"]

        # two particle-dependent factors: NO split
        v2 = pybamm.Variable("v2", domain="negative particle")
        out = pybamm.r_average(v_part * v2)
        assert isinstance(out, pybamm.RAverage)
        ravg_nodes = [n for n in out.pre_order() if isinstance(n, pybamm.RAverage)]
        assert len(ravg_nodes) == 1

    def test_size_average_does_not_factor_constants(self):
        # size_average is a weighted average with distribution-dependent weight
        # (f_a_dist). Factoring out constants would require regenerating f_a_dist
        # for each sub-expression, which can differ by domain and break conservation.
        # Therefore, size_average does NOT factor out size-constant operands.
        v_size = pybamm.Variable("v_size", domain="negative particle size")

        # All of these should remain as a single SizeAverage, not split
        assert isinstance(pybamm.size_average(2 * v_size), pybamm.SizeAverage)
        assert isinstance(pybamm.size_average(v_size * 2), pybamm.SizeAverage)
        assert isinstance(pybamm.size_average(v_size / 2), pybamm.SizeAverage)
        assert isinstance(pybamm.size_average(pybamm.t * v_size), pybamm.SizeAverage)

        # two size-dependent factors: also NO split
        v2 = pybamm.Variable("v2", domain="negative particle size")
        out = pybamm.size_average(v_size * v2)
        assert isinstance(out, pybamm.SizeAverage)

    def test_size_average_preserves_user_f_a_dist(self):
        # When the user supplies their own f_a_dist, the symbol must not be
        # split (the weight is tied to the parent symbol).
        v_size = pybamm.Variable("v_size", domain="negative particle size")
        R = pybamm.SpatialVariable("R", domains=v_size.domains, coord_sys="cartesian")
        custom_weight = pybamm.Scalar(3.0) * R
        out = pybamm.size_average(2 * v_size, f_a_dist=custom_weight)
        assert isinstance(out, pybamm.SizeAverage)
        assert out.f_a_dist == custom_weight

    def test_average_does_not_split_when_both_sides_depend_on_axis(self):
        """Sanity across all averages: with neither side constant, no split."""
        vcc1 = pybamm.Variable("a", domain="current collector")
        vcc2 = pybamm.Variable("b", domain="current collector")
        assert isinstance(pybamm.z_average(vcc1 / vcc2), pybamm.ZAverage)
        assert isinstance(pybamm.yz_average(vcc1 / vcc2), pybamm.YZAverage)

        vr1 = pybamm.Variable("vr1", domain="negative particle")
        vr2 = pybamm.Variable("vr2", domain="negative particle")
        assert isinstance(pybamm.r_average(vr1 / vr2), pybamm.RAverage)

        vs1 = pybamm.Variable("vs1", domain="negative particle size")
        vs2 = pybamm.Variable("vs2", domain="negative particle size")
        assert isinstance(pybamm.size_average(vs1 / vs2), pybamm.SizeAverage)

    def test_unary_new_copy_without_simplifications(self):
        def _check_copy(value, average_value):
            cls = type(average_value)
            copy = average_value._unary_new_copy(value, perform_simplifications=False)
            assert isinstance(copy, cls)

        # XAverage
        v_x = pybamm.Variable("v", domain="negative electrode")
        _check_copy(v_x, XAverage(v_x))

        # ZAverage
        v_z = pybamm.Variable("v", domain="current collector")
        _check_copy(v_z, ZAverage(v_z))

        # YZAverage
        _check_copy(v_z, YZAverage(v_z))

        # RAverage
        v_r = pybamm.Variable("v", domain="negative particle")
        _check_copy(v_r, RAverage(v_r))

        # SizeAverage
        v_size = pybamm.Variable("v", domain="negative particle size")
        f_a_dist = pybamm.Scalar(1.0)
        _check_copy(v_size, SizeAverage(v_size, f_a_dist))

    def test_size_average_domain_matches(self):
        assert SizeAverage.domain_matches("negative particle size")
        assert SizeAverage.domain_matches("positive particle size")
        assert SizeAverage.domain_matches("negative secondary particle size")
        assert not SizeAverage.domain_matches("negative electrode")
        assert not SizeAverage.domain_matches("negative particle")
