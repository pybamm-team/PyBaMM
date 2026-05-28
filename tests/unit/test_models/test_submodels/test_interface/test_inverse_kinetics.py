import numpy as np
import pytest

import pybamm
from pybamm.models.submodels.interface.kinetics.inverse_kinetics.base_inverse import (
    CurrentForInverseKinetics,
    CurrentForInverseKineticsLithiumMetal,
)
from pybamm.models.submodels.interface.kinetics.inverse_kinetics.inverse_butler_volmer import (
    InverseButlerVolmer,
)
from pybamm.models.submodels.interface.kinetics.inverse_kinetics.inverse_linear import (
    InverseLinear,
)


def _make_options(**kwargs):
    """Create options dict for inverse kinetics, with sensible defaults."""
    return {
        "SEI film resistance": "none",
        "total interfacial current density as a state": "false",
        "particle size": "single",
        "working electrode": "both",
        **kwargs,
    }


class TestInverseButlerVolmer:
    """Unit tests for InverseButlerVolmer._get_overpotential."""

    @pytest.fixture
    def model(self):
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        return InverseButlerVolmer(param, "negative", "lithium-ion main", options)

    def test_overpotential_formula(self, model):
        """Test that _get_overpotential returns the correct Butler-Volmer inversion."""
        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)

        R = pybamm.constants.R
        F = pybamm.constants.F
        expected = (2 * R * T / F / ne) * pybamm.arcsinh(j / (2 * j0 * u))

        assert eta.evaluate() == pytest.approx(expected.evaluate(), rel=1e-10)

    @pytest.mark.parametrize("j_val,expected_sign", [(5.0, 1), (-5.0, -1)])
    def test_overpotential_sign(self, model, j_val, expected_sign):
        """Overpotential sign should match current sign (arcsinh is odd)."""
        j = pybamm.Scalar(j_val)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)
        assert np.sign(eta.evaluate()) == expected_sign

    def test_overpotential_symmetry(self, model):
        """eta(-j) = -eta(j) due to odd symmetry of arcsinh."""
        j = pybamm.Scalar(3.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta_pos = model._get_overpotential(j, j0, ne, T, u)
        eta_neg = model._get_overpotential(-j, j0, ne, T, u)

        assert eta_pos.evaluate() == pytest.approx(-eta_neg.evaluate(), rel=1e-10)

    @pytest.mark.parametrize("zero_param", ["j0", "u"])
    def test_overpotential_finite_at_zero_denominator(self, model, zero_param):
        """arcsinh2 should return finite value when j0=0 or u=0."""
        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.0 if zero_param == "j0" else 0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(0.0 if zero_param == "u" else 1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)
        assert np.isfinite(eta.evaluate())

    def test_overpotential_ne_scaling(self, model):
        """Overpotential should scale as 1/ne."""
        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.5)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta_ne1 = model._get_overpotential(j, j0, pybamm.Scalar(1), T, u)
        eta_ne2 = model._get_overpotential(j, j0, pybamm.Scalar(2), T, u)

        assert eta_ne1.evaluate() == pytest.approx(2 * eta_ne2.evaluate(), rel=1e-10)

    def test_overpotential_T_scaling(self, model):
        """Overpotential should scale linearly with T."""
        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        u = pybamm.Scalar(1.0)

        eta1 = model._get_overpotential(j, j0, ne, pybamm.Scalar(300), u)
        eta2 = model._get_overpotential(j, j0, ne, pybamm.Scalar(350), u)

        assert eta2.evaluate() == pytest.approx(eta1.evaluate() * 350 / 300, rel=1e-10)

    def test_low_current_linear_regime(self, model):
        """At low j/j0, arcsinh(x) ~ x, so eta ~ (2RT/NeF) * j/(2j0u)."""
        j = pybamm.Scalar(0.001)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)

        R = pybamm.constants.R
        F = pybamm.constants.F
        eta_linear = (2 * R * T / F) * j / (2 * j0 * u)

        assert eta.evaluate() == pytest.approx(eta_linear.evaluate(), rel=1e-4)

    def test_high_current_log_regime(self, model):
        """At high j/j0, arcsinh(x) ~ ln(2x), so eta ~ (2RT/NeF) * ln(j/j0u)."""
        j = pybamm.Scalar(100.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)

        R = pybamm.constants.R
        F = pybamm.constants.F
        x = j / (2 * j0 * u)
        eta_log = (2 * R * T / F / ne) * pybamm.log(2 * x)

        assert eta.evaluate() == pytest.approx(eta_log.evaluate(), rel=1e-2)


class TestInverseLinear:
    """Unit tests for InverseLinear._get_overpotential."""

    @pytest.fixture
    def model(self):
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        return InverseLinear(param, "negative", "lithium-ion main", options)

    def test_overpotential_formula(self, model):
        """Test linear approximation: eta = (2RT/NeF) * j/(2j0u)."""
        j = pybamm.Scalar(0.01)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)

        R = pybamm.constants.R
        F = pybamm.constants.F
        expected = (2 * R * T / F / ne) * j / (2 * j0 * u)

        assert eta.evaluate() == pytest.approx(expected.evaluate(), rel=1e-10)

    @pytest.mark.parametrize("zero_param", ["j0", "u"])
    def test_division_by_zero(self, model, zero_param):
        """InverseLinear should raise ZeroDivisionError when j0=0 or u=0."""
        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.0 if zero_param == "j0" else 0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(0.0 if zero_param == "u" else 1.0)

        with pytest.raises(ZeroDivisionError):
            eta = model._get_overpotential(j, j0, ne, T, u)
            eta.evaluate()

    def test_agrees_with_bv_at_low_current(self, model):
        """InverseLinear should agree with InverseButlerVolmer at low j/j0."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        bv_model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(0.001)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta_bv = bv_model._get_overpotential(j, j0, ne, T, u)
        eta_lin = model._get_overpotential(j, j0, ne, T, u)

        assert eta_lin.evaluate() == pytest.approx(eta_bv.evaluate(), rel=1e-3)


class TestCurrentForInverseKinetics:
    """Tests for CurrentForInverseKinetics submodel."""

    @pytest.mark.parametrize(
        "j_tot,j_sei,j_strip,expected",
        [
            (10.0, 0.0, 0.0, 10.0),
            (10.0, 1.0, 0.5, 8.5),
            (10.0, 2.0, 3.0, 5.0),
            (0.0, 0.0, 0.0, 0.0),
            (-5.0, 1.0, 1.0, -7.0),
        ],
    )
    def test_current_formula(self, j_tot, j_sei, j_strip, expected):
        """Test j = j_tot - j_sei - j_stripping computation."""
        from unittest.mock import patch

        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = CurrentForInverseKinetics(
            param, "negative", "lithium-ion main", options
        )

        variables = {
            "X-averaged negative electrode total interfacial current density [A.m-2]": pybamm.Scalar(
                j_tot
            ),
            "Negative electrode SEI interfacial current density [A.m-2]": pybamm.Scalar(
                j_sei
            ),
            "Negative electrode lithium plating interfacial current density [A.m-2]": pybamm.Scalar(
                j_strip
            ),
        }

        with (
            patch.object(
                model, "_get_standard_interfacial_current_variables"
            ) as mock_std,
            patch.object(
                model, "_get_standard_volumetric_current_density_variables"
            ) as mock_vol,
        ):
            mock_std.return_value = {}
            mock_vol.return_value = {}

            model.get_coupled_variables(variables)

            j_computed = mock_std.call_args[0][0]
            assert j_computed.evaluate() == pytest.approx(expected, rel=1e-10)


class TestCurrentForInverseKineticsLithiumMetal:
    """Tests for CurrentForInverseKineticsLithiumMetal submodel."""

    @pytest.mark.parametrize("i_cc", [-10.0, -1.0, 0.0, 1.0, 10.0])
    def test_current_equals_boundary(self, i_cc):
        """j should equal i_boundary_cc for lithium metal."""
        param = pybamm.LithiumIonParameters()
        options = _make_options(**{"working electrode": "positive"})
        model = CurrentForInverseKineticsLithiumMetal(
            param, "negative", "lithium metal plating", options
        )

        variables = {
            "Current collector current density [A.m-2]": pybamm.Scalar(i_cc),
        }
        result = model.get_coupled_variables(variables)
        j = result["Lithium metal plating current density [A.m-2]"]

        assert j.evaluate() == pytest.approx(i_cc, rel=1e-10)
