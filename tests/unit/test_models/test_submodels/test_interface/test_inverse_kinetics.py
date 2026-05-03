from unittest.mock import patch

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
    options = {
        "SEI film resistance": "none",
        "total interfacial current density as a state": "false",
        "particle size": "single",
        "working electrode": "both",
        **kwargs,
    }
    return options


class TestInverseButlerVolmer:
    """Unit tests for InverseButlerVolmer._get_overpotential."""

    def test_creation(self):
        """Test that InverseButlerVolmer can be instantiated."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)
        assert model is not None
        assert model.domain == "negative"
        assert model.reaction == "lithium-ion main"

    def test_overpotential_formula(self):
        """Test that _get_overpotential returns the correct Butler-Volmer inversion.

        eta = (2 * R * T / F / ne) * arcsinh(j / (2 * j0 * u))
        """
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)

        R = pybamm.constants.R
        F = pybamm.constants.F
        expected = (2 * R * T / F / ne) * pybamm.arcsinh(j / (2 * j0 * u))

        expected_val = expected.evaluate()
        actual_val = eta.evaluate()

        assert actual_val == pytest.approx(expected_val, rel=1e-10)

    def test_overpotential_sign_positive_current(self):
        """Positive j should give positive overpotential."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(5.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)
        assert eta.evaluate() > 0

    def test_overpotential_sign_negative_current(self):
        """Negative j should give negative overpotential."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(-5.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)
        assert eta.evaluate() < 0

    def test_overpotential_symmetry(self):
        """eta(-j) = -eta(j) due to odd symmetry of arcsinh."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(3.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta_pos = model._get_overpotential(j, j0, ne, T, u)
        eta_neg = model._get_overpotential(-j, j0, ne, T, u)

        assert eta_pos.evaluate() == pytest.approx(-eta_neg.evaluate(), rel=1e-10)

    def test_overpotential_j0_zero_finite(self):
        """arcsinh2 should return finite value when j0 = 0."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.0)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)
        result = eta.evaluate()

        assert np.isfinite(result), f"eta should be finite when j0=0, got {result}"

    def test_overpotential_utilisation_zero_finite(self):
        """arcsinh2 should return finite value when u = 0."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(0.0)

        eta = model._get_overpotential(j, j0, ne, T, u)
        result = eta.evaluate()

        assert np.isfinite(result), f"eta should be finite when u=0, got {result}"

    def test_overpotential_ne_dependency(self):
        """Overpotential should scale as 1/ne."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.5)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta_ne1 = model._get_overpotential(j, j0, pybamm.Scalar(1), T, u)
        eta_ne2 = model._get_overpotential(j, j0, pybamm.Scalar(2), T, u)

        assert eta_ne1.evaluate() == pytest.approx(2 * eta_ne2.evaluate(), rel=1e-10)

    def test_overpotential_T_linearity(self):
        """Overpotential should scale linearly with T."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        u = pybamm.Scalar(1.0)

        T1 = pybamm.Scalar(300)
        T2 = pybamm.Scalar(350)

        eta1 = model._get_overpotential(j, j0, ne, T1, u)
        eta2 = model._get_overpotential(j, j0, ne, T2, u)

        assert eta2.evaluate() == pytest.approx(eta1.evaluate() * 350 / 300, rel=1e-10)

    def test_overpotential_low_j_linear_regime(self):
        """At low j/j0, arcsinh(x) ~ x, so eta ~ (2RT/NeF) * j/(2j0u)."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

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

    def test_overpotential_high_j_log_regime(self):
        """At high j/j0, arcsinh(x) ~ ln(2x), so eta ~ (2RT/NeF) * ln(j/j0)."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(100.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)

        R = pybamm.constants.R
        F = pybamm.constants.F
        eta_log = (2 * R * T / F) * pybamm.arcsinh(j / (2 * j0 * u))

        assert eta.evaluate() == pytest.approx(eta_log.evaluate(), rel=1e-10)

    def test_overpotential_with_input_parameters(self):
        """_get_overpotential should work with InputParameters."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        j = pybamm.InputParameter("j")
        j0 = pybamm.InputParameter("j0")
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)

        val = eta.evaluate(inputs={"j": 1.0, "j0": 0.5})
        expected = (
            2 * pybamm.constants.R * T / pybamm.constants.F / ne
        ) * pybamm.arcsinh(1.0 / (2 * 0.5 * 1.0))
        assert val == pytest.approx(expected.evaluate(), rel=1e-10)

    def test_overpotential_returns_multiplication_node(self):
        """The result should be a Multiplication node (prefactor * arcsinh2)."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)

        assert isinstance(eta, pybamm.Multiplication)

    def test_overpotential_with_broadcast_j(self):
        """_get_overpotential should handle broadcast j values."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)

        c_e = pybamm.Variable(
            "concentration [mol.m-3]",
            domain=["negative electrode"],
            auxiliary_domains={"secondary": "current collector"},
        )
        j = pybamm.Scalar(1.0) + 0 * c_e
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)

        assert isinstance(eta, pybamm.Multiplication)


class TestInverseLinear:
    """Unit tests for InverseLinear._get_overpotential."""

    def test_creation(self):
        """Test that InverseLinear can be instantiated."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseLinear(param, "negative", "lithium-ion main", options)
        assert model is not None
        assert model.domain == "negative"

    def test_overpotential_formula(self):
        """Test that _get_overpotential returns the linear approximation.

        eta = (2 * R * T / F / ne) * j / (2 * j0 * u)
        """
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseLinear(param, "negative", "lithium-ion main", options)

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

    def test_overpotential_division_by_zero_j0(self):
        """InverseLinear should raise ZeroDivisionError when j0 = 0."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseLinear(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.0)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        with pytest.raises(ZeroDivisionError):
            eta = model._get_overpotential(j, j0, ne, T, u)
            eta.evaluate()

    def test_overpotential_division_by_zero_utilisation(self):
        """InverseLinear should raise ZeroDivisionError when u = 0."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseLinear(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(1.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(0.0)

        with pytest.raises(ZeroDivisionError):
            eta = model._get_overpotential(j, j0, ne, T, u)
            eta.evaluate()

    def test_overpotential_sign(self):
        """Negative j should give negative overpotential."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseLinear(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(-1.0)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta = model._get_overpotential(j, j0, ne, T, u)
        assert eta.evaluate() < 0

    def test_overpotential_ne_dependency(self):
        """Overpotential should scale as 1/ne."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseLinear(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(0.01)
        j0 = pybamm.Scalar(0.5)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta_ne1 = model._get_overpotential(j, j0, pybamm.Scalar(1), T, u)
        eta_ne2 = model._get_overpotential(j, j0, pybamm.Scalar(2), T, u)

        assert eta_ne1.evaluate() == pytest.approx(2 * eta_ne2.evaluate(), rel=1e-10)

    def test_overpotential_T_linearity(self):
        """Overpotential should scale linearly with T."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = InverseLinear(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(0.01)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        u = pybamm.Scalar(1.0)

        T1 = pybamm.Scalar(300)
        T2 = pybamm.Scalar(350)

        eta1 = model._get_overpotential(j, j0, ne, T1, u)
        eta2 = model._get_overpotential(j, j0, ne, T2, u)

        assert eta2.evaluate() == pytest.approx(eta1.evaluate() * 350 / 300, rel=1e-10)

    def test_inverse_vs_linear_agreement_at_low_current(self):
        """InverseLinear should agree with InverseButlerVolmer at low j/j0."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()

        bv_model = InverseButlerVolmer(param, "negative", "lithium-ion main", options)
        lin_model = InverseLinear(param, "negative", "lithium-ion main", options)

        j = pybamm.Scalar(0.001)
        j0 = pybamm.Scalar(0.5)
        ne = pybamm.Scalar(1)
        T = pybamm.Scalar(300)
        u = pybamm.Scalar(1.0)

        eta_bv = bv_model._get_overpotential(j, j0, ne, T, u)
        eta_lin = lin_model._get_overpotential(j, j0, ne, T, u)

        assert eta_lin.evaluate() == pytest.approx(eta_bv.evaluate(), rel=1e-3)


class TestCurrentForInverseKinetics:
    """Tests for CurrentForInverseKinetics submodel."""

    def test_creation(self):
        """Test that CurrentForInverseKinetics can be instantiated."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = CurrentForInverseKinetics(
            param, "negative", "lithium-ion main", options
        )
        assert model is not None

    def test_current_formula(self):
        """Test j = j_tot - j_sei - j_stripping computation."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = CurrentForInverseKinetics(
            param, "negative", "lithium-ion main", options
        )

        j_tot = pybamm.Scalar(10.0)
        j_sei = pybamm.Scalar(1.0)
        j_stripping = pybamm.Scalar(0.5)

        variables = {
            "X-averaged negative electrode total interfacial current density [A.m-2]": j_tot,
            "Negative electrode SEI interfacial current density [A.m-2]": j_sei,
            "Negative electrode lithium plating interfacial current density [A.m-2]": j_stripping,
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

            # Verify j was computed correctly and passed to helper
            mock_std.assert_called_once()
            j_computed = mock_std.call_args[0][0]
            assert j_computed.evaluate() == pytest.approx(8.5, rel=1e-10)

    def test_current_formula_various_values(self):
        """Test j = j_tot - j_sei - j_stripping with various inputs."""
        param = pybamm.LithiumIonParameters()
        options = _make_options()
        model = CurrentForInverseKinetics(
            param, "negative", "lithium-ion main", options
        )

        test_cases = [
            (10.0, 0.0, 0.0, 10.0),
            (10.0, 2.0, 3.0, 5.0),
            (5.0, 1.0, 1.0, 3.0),
            (0.0, 0.0, 0.0, 0.0),
            (-5.0, 1.0, 1.0, -7.0),
        ]

        for j_tot_val, j_sei_val, j_strip_val, expected in test_cases:
            variables = {
                "X-averaged negative electrode total interfacial current density [A.m-2]": pybamm.Scalar(
                    j_tot_val
                ),
                "Negative electrode SEI interfacial current density [A.m-2]": pybamm.Scalar(
                    j_sei_val
                ),
                "Negative electrode lithium plating interfacial current density [A.m-2]": pybamm.Scalar(
                    j_strip_val
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
                assert j_computed.evaluate() == pytest.approx(expected, rel=1e-10), (
                    f"Failed for inputs ({j_tot_val}, {j_sei_val}, {j_strip_val})"
                )


class TestCurrentForInverseKineticsLithiumMetal:
    """Tests for CurrentForInverseKineticsLithiumMetal submodel."""

    def test_creation(self):
        """Test that CurrentForInverseKineticsLithiumMetal can be instantiated."""
        param = pybamm.LithiumIonParameters()
        options = _make_options(**{"working electrode": "positive"})
        model = CurrentForInverseKineticsLithiumMetal(
            param, "negative", "lithium metal plating", options
        )
        assert model is not None

    def test_current_equals_boundary(self):
        """j should equal i_boundary_cc for lithium metal."""
        param = pybamm.LithiumIonParameters()
        options = _make_options(**{"working electrode": "positive"})
        model = CurrentForInverseKineticsLithiumMetal(
            param, "negative", "lithium metal plating", options
        )

        variables = {
            "Current collector current density [A.m-2]": pybamm.Scalar(5.0),
        }

        result = model.get_coupled_variables(variables)
        j = result["Lithium metal plating current density [A.m-2]"]

        assert j.evaluate() == pytest.approx(5.0, rel=1e-10)

    def test_current_with_different_boundary_values(self):
        """j should track the boundary current for any value."""
        param = pybamm.LithiumIonParameters()
        options = _make_options(**{"working electrode": "positive"})
        model = CurrentForInverseKineticsLithiumMetal(
            param, "negative", "lithium metal plating", options
        )

        for i_cc in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            variables = {
                "Current collector current density [A.m-2]": pybamm.Scalar(i_cc),
            }
            result = model.get_coupled_variables(variables)
            j = result["Lithium metal plating current density [A.m-2]"]
            assert j.evaluate() == pytest.approx(i_cc, rel=1e-10)
