#
# Tests for the PE phase-transition core-shell degradation submodels
#
import pytest

import pybamm


@pytest.fixture
def positive_pe_degradation_setup():
    """Build a (param, options) pair valid for the PE phase-transition submodels."""
    options = pybamm.BatteryModelOptions({"PE degradation": "phase transition"})
    param = pybamm.LithiumIonParameters(options)
    return param, options


class TestPhaseTransition:
    def test_domain_must_be_positive(self):
        # The PhaseTransition submodel only models the positive electrode.
        with pytest.raises(pybamm.DomainError, match=r"Domain must be 'positive'"):
            pybamm.pe_degradation.PhaseTransition(None, "negative", {})

    def test_x_average_flag(self, positive_pe_degradation_setup):
        param, options = positive_pe_degradation_setup
        sub_dist = pybamm.pe_degradation.PhaseTransition(
            param, "positive", options, x_average=False
        )
        sub_xav = pybamm.pe_degradation.PhaseTransition(
            param, "positive", options, x_average=True
        )
        assert sub_dist.x_average is False
        assert sub_xav.x_average is True


class TestBasePhaseTransition:
    def test_r_average_shell_wrong_domain_raises(self, positive_pe_degradation_setup):
        param, options = positive_pe_degradation_setup
        sub = pybamm.pe_degradation.PhaseTransition(param, "positive", options)
        # Symbol whose domain is not a 'shell' domain
        a = pybamm.Symbol("a", domain=["positive core"])
        s_nd = pybamm.Scalar(0.5)
        with pytest.raises(pybamm.DomainError, match=r"shell"):
            sub._r_average_shell(a, s_nd)


class TestTotalConcentration:
    def test_instantiates(self, positive_pe_degradation_setup):
        param, options = positive_pe_degradation_setup
        sub = pybamm.pe_degradation.TotalConcentration(param, "positive", options)
        assert sub.domain == "positive"


class TestInverseButlerVolmerPeShellHook:
    """The InverseButlerVolmer override of _get_pe_shell_potential_drop adds the
    PE phase-transition shell-layer drop to the SPM/SPMe inverse kinetics path."""

    def test_negative_domain_returns_zero(self):
        # The hook must return Scalar(0) for the negative electrode regardless
        # of what the PE degradation option says.
        options = pybamm.BatteryModelOptions({"PE degradation": "phase transition"})
        param = pybamm.LithiumIonParameters(options)
        sub = pybamm.kinetics.InverseButlerVolmer(
            param, "negative", "lithium-ion main", options
        )
        j_tot = pybamm.Scalar(1.0)
        variables: dict = {}
        result = sub._get_pe_shell_potential_drop(j_tot, variables)
        assert isinstance(result, pybamm.Scalar)
        assert result.value == 0
        # No PE shell variables should be injected
        assert variables == {}

    def test_pe_degradation_off_returns_zero(self):
        # On the positive electrode but with PE degradation disabled, returns 0.
        options = pybamm.BatteryModelOptions({})  # default: PE degradation = "none"
        param = pybamm.LithiumIonParameters(options)
        sub = pybamm.kinetics.InverseButlerVolmer(
            param, "positive", "lithium-ion main", options
        )
        j_tot = pybamm.Scalar(1.0)
        variables: dict = {}
        result = sub._get_pe_shell_potential_drop(j_tot, variables)
        assert isinstance(result, pybamm.Scalar)
        assert result.value == 0
        assert variables == {}

    def test_phase_transition_returns_nonzero_and_updates_variables(self):
        # On the positive electrode with PE phase transition on, returns a
        # non-trivial eta_shell expression and registers PE shell overpotential
        # variables.
        options = pybamm.BatteryModelOptions({"PE degradation": "phase transition"})
        param = pybamm.LithiumIonParameters(options)
        sub = pybamm.kinetics.InverseButlerVolmer(
            param, "positive", "lithium-ion main", options
        )
        j_tot = pybamm.Scalar(1.0)
        variables: dict = {
            "X-averaged positive particle moving phase boundary location": (
                pybamm.Scalar(0.9)
            ),
            "X-averaged positive particle radius [m]": pybamm.Scalar(3.8e-6),
        }
        result = sub._get_pe_shell_potential_drop(j_tot, variables)
        # Non-trivial expression — not just Scalar(0)
        assert not (isinstance(result, pybamm.Scalar) and result.value == 0)
        # The hook should have injected the standard PE shell overpotential vars
        new_keys = set(variables) - {
            "X-averaged positive particle moving phase boundary location",
            "X-averaged positive particle radius [m]",
        }
        assert len(new_keys) > 0


class TestPhaseTransitionFundamentalVariables:
    """The fundamental-variable contract: the keys produced here are consumed by
    the inverse Butler-Volmer hook and the base kinetics PE-shell branch."""

    @pytest.mark.parametrize("x_average", [False, True])
    def test_fundamental_variables_register_expected_keys(
        self, positive_pe_degradation_setup, x_average
    ):
        param, options = positive_pe_degradation_setup
        sub = pybamm.pe_degradation.PhaseTransition(
            param, "positive", options, x_average=x_average
        )
        variables = sub.get_fundamental_variables()
        # Keys consumed by InverseButlerVolmer._get_pe_shell_potential_drop and
        # by base_kinetics.py PE-shell branch
        assert (
            "X-averaged positive particle moving phase boundary location" in variables
        )
        assert "Positive particle moving phase boundary location" in variables
        # Keys for shell oxygen and core lithium concentrations
        assert "Positive core lithium concentration [mol.m-3]" in variables
        assert "Positive shell oxygen concentration [mol.m-3]" in variables
        # LAM diagnostic
        assert (
            "X-averaged loss of active material due to PE phase transition" in variables
        )
