#
# Test the experiment step termination classes
#

from types import SimpleNamespace

import pytest

import pybamm


class TestExperimentStepTermination:
    def test_base_termination(self):
        term = pybamm.step.BaseTermination(1)
        assert term.value == 1
        with pytest.raises(NotImplementedError):
            term.get_event(None, None)
        assert term == pybamm.step.BaseTermination(1)
        assert term != pybamm.step.BaseTermination(2)
        assert term != pybamm.step.CurrentTermination(1)

    def test_c_rate_termination(self):
        term = pybamm.step.CRateTermination(0.02)
        assert term.value == 0.02
        assert term.operator is None
        variables = {"C-rate": pybamm.Scalar(0.02)}
        assert term.get_event(variables, None).evaluate() == 0
        with pytest.warns(DeprecationWarning):
            term_old = pybamm.step.CrateTermination(0.02)
            assert (
                term.get_event(variables, None).evaluate()
                == term_old.get_event(variables, None).evaluate()
            )

    def test_voltage_termination_returns_none_without_charge_or_discharge(self):
        term = pybamm.step.VoltageTermination(4.2)
        step = SimpleNamespace(direction="rest")
        variables = {"Battery voltage [V]": pybamm.Scalar(4.2)}

        assert term.get_event_name(step) is None
        assert term.get_event_expression(variables, step) is None
        assert term.get_event(variables, step) is None
