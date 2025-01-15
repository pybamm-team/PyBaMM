#
# Tests for the Broadcast class
#
import pytest
from tests import assert_domain_equal
import numpy as np

import pybamm


class TestBroadcasts:
    def test_primary_broadcast(self):
        a = pybamm.Symbol("a")
        broad_a = pybamm.PrimaryBroadcast(a, ["negative electrode"])
        assert broad_a.name == "broadcast"
        assert broad_a.children[0].name == a.name
        assert broad_a.domain == ["negative electrode"]
        assert broad_a.broadcasts_to_nodes
        assert broad_a.reduce_one_dimension() == a

        a = pybamm.Symbol(
            "a",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        broad_a = pybamm.PrimaryBroadcast(a, ["negative particle"])
        assert_domain_equal(
            broad_a.domains,
            {
                "primary": ["negative particle"],
                "secondary": ["negative electrode"],
                "tertiary": ["current collector"],
            },
        )
        a = pybamm.Symbol(
            "a",
            domain="negative particle size",
            auxiliary_domains={
                "secondary": "negative electrode",
                "tertiary": "current collector",
            },
        )
        broad_a = pybamm.PrimaryBroadcast(a, ["negative particle"])
        assert_domain_equal(
            broad_a.domains,
            {
                "primary": ["negative particle"],
                "secondary": ["negative particle size"],
                "tertiary": ["negative electrode"],
                "quaternary": ["current collector"],
            },
        )

        a = pybamm.Symbol("a", domain="current collector")
        with pytest.raises(
            pybamm.DomainError, match="Cannot Broadcast an object into empty domain"
        ):
            pybamm.PrimaryBroadcast(a, [])
        with pytest.raises(
            pybamm.DomainError, match="Primary broadcast from current collector"
        ):
            pybamm.PrimaryBroadcast(a, "bad domain")
        a = pybamm.Symbol("a", domain="negative electrode")
        with pytest.raises(
            pybamm.DomainError, match="Primary broadcast from electrode"
        ):
            pybamm.PrimaryBroadcast(a, "current collector")
        a = pybamm.Symbol("a", domain="negative particle size")
        with pytest.raises(
            pybamm.DomainError, match="Primary broadcast from particle size"
        ):
            pybamm.PrimaryBroadcast(a, "negative electrode")
        a = pybamm.Symbol("a", domain="negative particle")
        with pytest.raises(
            pybamm.DomainError, match="Cannot do primary broadcast from particle domain"
        ):
            pybamm.PrimaryBroadcast(a, "current collector")

    def test_secondary_broadcast(self):
        a = pybamm.Symbol(
            "a",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "current collector"},
        )
        broad_a = pybamm.SecondaryBroadcast(a, ["negative electrode"])
        assert_domain_equal(
            broad_a.domains,
            {
                "primary": ["negative particle"],
                "secondary": ["negative electrode"],
                "tertiary": ["current collector"],
            },
        )
        assert broad_a.broadcasts_to_nodes
        broadbroad_a = pybamm.SecondaryBroadcast(broad_a, ["negative particle size"])
        assert_domain_equal(
            broadbroad_a.domains,
            {
                "primary": ["negative particle"],
                "secondary": ["negative particle size"],
                "tertiary": ["negative electrode"],
                "quaternary": ["current collector"],
            },
        )

        assert broad_a.reduce_one_dimension() == a

        a = pybamm.Symbol("a")
        with pytest.raises(TypeError, match="empty domain"):
            pybamm.SecondaryBroadcast(a, "current collector")
        a = pybamm.Symbol("a", domain="negative particle")
        with pytest.raises(
            pybamm.DomainError, match="Secondary broadcast from particle"
        ):
            pybamm.SecondaryBroadcast(a, "negative particle")
        a = pybamm.Symbol("a", domain="negative particle size")
        with pytest.raises(
            pybamm.DomainError, match="Secondary broadcast from particle size"
        ):
            pybamm.SecondaryBroadcast(a, "negative particle")
        a = pybamm.Symbol("a", domain="negative electrode")
        with pytest.raises(
            pybamm.DomainError, match="Secondary broadcast from electrode"
        ):
            pybamm.SecondaryBroadcast(a, "negative particle")

        a = pybamm.Symbol("a", domain="current collector")
        with pytest.raises(pybamm.DomainError, match="Cannot do secondary broadcast"):
            pybamm.SecondaryBroadcast(a, "electrode")

    def test_tertiary_broadcast(self):
        a = pybamm.Symbol(
            "a",
            domain=["negative particle"],
            auxiliary_domains={
                "secondary": "negative particle size",
                "tertiary": "current collector",
            },
        )
        broad_a = pybamm.TertiaryBroadcast(a, "negative electrode")
        assert_domain_equal(
            broad_a.domains,
            {
                "primary": ["negative particle"],
                "secondary": ["negative particle size"],
                "tertiary": ["negative electrode"],
                "quaternary": ["current collector"],
            },
        )
        assert broad_a.broadcasts_to_nodes

        with pytest.raises(NotImplementedError):
            broad_a.reduce_one_dimension()

        a_no_secondary = pybamm.Symbol("a", domain="negative particle")
        with pytest.raises(TypeError, match="without a secondary"):
            pybamm.TertiaryBroadcast(a_no_secondary, "negative electrode")
        with pytest.raises(
            pybamm.DomainError, match="Tertiary broadcast from a symbol with particle"
        ):
            pybamm.TertiaryBroadcast(a, "negative particle")
        a = pybamm.Symbol(
            "a",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "negative electrode"},
        )
        with pytest.raises(
            pybamm.DomainError,
            match="Tertiary broadcast from a symbol with an electrode",
        ):
            pybamm.TertiaryBroadcast(a, "negative particle size")
        a = pybamm.Symbol(
            "a",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "current collector"},
        )
        with pytest.raises(pybamm.DomainError, match="Cannot do tertiary broadcast"):
            pybamm.TertiaryBroadcast(a, "negative electrode")

    def test_full_broadcast(self):
        a = pybamm.Symbol("a")
        broad_a = pybamm.FullBroadcast(a, ["negative electrode"], "current collector")
        assert broad_a.domain == ["negative electrode"]
        assert broad_a.domains["secondary"] == ["current collector"]
        assert broad_a.broadcasts_to_nodes
        assert broad_a.reduce_one_dimension() == pybamm.PrimaryBroadcast(
            a, "current collector"
        )

        broad_a = pybamm.FullBroadcast(a, ["negative electrode"], {})
        assert broad_a.reduce_one_dimension() == a

        broad_a = pybamm.FullBroadcast(
            a,
            "negative particle",
            {
                "secondary": "negative particle size",
                "tertiary": "negative electrode",
                "quaternary": "current collector",
            },
        )
        assert broad_a.reduce_one_dimension() == pybamm.FullBroadcast(
            a,
            "negative particle size",
            {
                "secondary": "negative electrode",
                "tertiary": "current collector",
            },
        )

        with pytest.raises(
            pybamm.DomainError,
            match="Cannot do full broadcast to an empty primary domain",
        ):
            pybamm.FullBroadcast(a, [])

    def test_full_broadcast_number(self):
        broad_a = pybamm.FullBroadcast(1, ["negative electrode"], None)
        assert broad_a.name == "broadcast"
        assert isinstance(broad_a.children[0], pybamm.Symbol)
        assert broad_a.children[0].evaluate() == np.array([1])
        assert broad_a.domain == ["negative electrode"]

        a = pybamm.Symbol("a", domain="current collector")
        with pytest.raises(pybamm.DomainError, match="Cannot do full broadcast"):
            pybamm.FullBroadcast(a, "electrode", None)

    def test_ones_like(self):
        a = pybamm.Parameter("a")
        ones_like_a = pybamm.ones_like(a)
        assert ones_like_a == pybamm.Scalar(1)

        a = pybamm.Variable(
            "a",
            domain="negative electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        ones_like_a = pybamm.ones_like(a)
        assert isinstance(ones_like_a, pybamm.FullBroadcast)
        assert ones_like_a.name == "broadcast"
        assert ones_like_a.domains == a.domains

        b = pybamm.Variable("b", domain="current collector")
        ones_like_ab = pybamm.ones_like(b, a)
        assert isinstance(ones_like_ab, pybamm.FullBroadcast)
        assert ones_like_ab.name == "broadcast"
        assert ones_like_ab.domains == a.domains

    def test_broadcast_to_edges(self):
        a = pybamm.Symbol("a")

        # primary
        broad_a = pybamm.PrimaryBroadcastToEdges(a, ["negative electrode"])
        assert broad_a.name == "broadcast to edges"
        assert broad_a.children[0].name == a.name
        assert broad_a.domain == ["negative electrode"]
        assert broad_a.evaluates_on_edges("primary")
        assert not broad_a.broadcasts_to_nodes
        assert broad_a.reduce_one_dimension() == a

        # secondary
        a = pybamm.Symbol(
            "a",
            domain=["negative particle"],
            auxiliary_domains={"secondary": "current collector"},
        )
        broad_a = pybamm.SecondaryBroadcastToEdges(a, ["negative electrode"])
        assert_domain_equal(
            broad_a.domains,
            {
                "primary": ["negative particle"],
                "secondary": ["negative electrode"],
                "tertiary": ["current collector"],
            },
        )
        assert broad_a.evaluates_on_edges("primary")
        assert not broad_a.broadcasts_to_nodes

        # tertiary
        a = pybamm.Symbol(
            "a",
            domain=["negative particle"],
            auxiliary_domains={
                "secondary": "negative particle size",
                "tertiary": "current collector",
            },
        )
        broad_a = pybamm.TertiaryBroadcastToEdges(a, ["negative electrode"])
        assert_domain_equal(
            broad_a.domains,
            {
                "primary": ["negative particle"],
                "secondary": ["negative particle size"],
                "tertiary": ["negative electrode"],
                "quaternary": ["current collector"],
            },
        )
        assert broad_a.evaluates_on_edges("primary")
        assert not broad_a.broadcasts_to_nodes

        # full
        a = pybamm.Symbol("a")
        broad_a = pybamm.FullBroadcastToEdges(
            a, ["negative electrode"], "current collector"
        )
        assert broad_a.domain == ["negative electrode"]
        assert broad_a.domains["secondary"] == ["current collector"]
        assert broad_a.evaluates_on_edges("primary")
        assert not broad_a.broadcasts_to_nodes
        assert broad_a.reduce_one_dimension() == pybamm.PrimaryBroadcastToEdges(
            a, "current collector"
        )
        broad_a = pybamm.FullBroadcastToEdges(a, ["negative electrode"], {})
        assert broad_a.reduce_one_dimension() == a

        broad_a = pybamm.FullBroadcastToEdges(
            a,
            "negative particle",
            {"secondary": "negative electrode", "tertiary": "current collector"},
        )
        assert broad_a.reduce_one_dimension() == pybamm.FullBroadcastToEdges(
            a, "negative electrode", "current collector"
        )

    def test_to_equation(self):
        a = pybamm.PrimaryBroadcast(0, "test").to_equation()
        assert a == 0

    def test_diff(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.PrimaryBroadcast(a, "separator")
        y = np.array([5])
        # diff of broadcast is broadcast of diff
        d = b.diff(a)
        assert isinstance(d, pybamm.PrimaryBroadcast)
        assert d.child.evaluate(y=y) == 1
        # diff of broadcast w.r.t. itself is 1
        d = b.diff(b)
        assert isinstance(d, pybamm.Scalar)
        assert d.evaluate(y=y) == 1
        # diff of broadcast of a constant is 0
        c = pybamm.PrimaryBroadcast(pybamm.Scalar(4), "separator")
        d = c.diff(a)
        assert isinstance(d, pybamm.Scalar)
        assert d.evaluate(y=y) == 0

    def test_to_from_json_error(self):
        a = pybamm.StateVector(slice(0, 1))
        b = pybamm.PrimaryBroadcast(a, "separator")
        with pytest.raises(NotImplementedError):
            b.to_json()

        with pytest.raises(NotImplementedError):
            pybamm.PrimaryBroadcast._from_json({})
