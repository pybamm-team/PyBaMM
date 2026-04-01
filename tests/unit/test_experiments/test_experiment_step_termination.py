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

    def test_base_termination_invalid_operator(self):
        with pytest.raises(ValueError, match="Invalid operator"):
            pybamm.step.BaseTermination(1, operator="=")

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

    def test_current_and_voltage_termination_operator_branches(self):
        variables = {
            "Current [A]": pybamm.Scalar(0.5),
            "Battery voltage [V]": pybamm.Scalar(4.2),
        }
        rest_step = SimpleNamespace(direction="rest")

        current_gt = pybamm.step.CurrentTermination(0.4, operator=">")
        current_lt = pybamm.step.CurrentTermination(0.6, operator="<")
        assert current_gt.get_event_name(None) == "Current [A] > 0.4 [A] [experiment]"
        assert current_lt.get_event_name(None) == "Current [A] < 0.6 [A] [experiment]"
        assert current_gt.get_event_expression(
            variables, None
        ).evaluate() == pytest.approx(-0.1)
        assert current_lt.get_event_expression(
            variables, None
        ).evaluate() == pytest.approx(-0.1)

        voltage_gt = pybamm.step.VoltageTermination(4.1, operator=">")
        voltage_lt = pybamm.step.VoltageTermination(4.3, operator="<")
        assert voltage_gt.get_event_name(rest_step) == "Voltage > 4.1 [V] [experiment]"
        assert voltage_lt.get_event_name(rest_step) == "Voltage < 4.3 [V] [experiment]"
        assert voltage_gt.get_event_expression(
            variables, rest_step
        ).evaluate() == pytest.approx(-0.1)
        assert voltage_lt.get_event_expression(
            variables, rest_step
        ).evaluate() == pytest.approx(-0.1)

        assert (pybamm.step.step_termination.Current() > 0.4) == current_gt
        assert (pybamm.step.step_termination.Current() < 0.6) == current_lt
        assert (pybamm.step.step_termination.Voltage() > 4.1) == voltage_gt
        assert (pybamm.step.step_termination.Voltage() < 4.3) == voltage_lt

    def test_voltage_termination_returns_none_without_charge_or_discharge(self):
        term = pybamm.step.VoltageTermination(4.2)
        step = SimpleNamespace(direction="rest")
        variables = {"Battery voltage [V]": pybamm.Scalar(4.2)}

        assert term.get_event_name(step) is None
        assert term.get_event_expression(variables, step) is None
        assert term.get_event(variables, step) is None

    def test_read_termination_tuple(self):
        current = pybamm.step.step_termination._read_termination((">", "current", 2.0))
        voltage = pybamm.step.step_termination._read_termination(("<", "voltage", 3.0))
        crate = pybamm.step.step_termination._read_termination((">", "C-rate", 0.5))

        assert current == pybamm.step.CurrentTermination(2.0, operator=">")
        assert voltage == pybamm.step.VoltageTermination(3.0, operator="<")
        assert crate == pybamm.step.CRateTermination(0.5, operator=">")
