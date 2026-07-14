#
# Test the experiment step termination classes
#

import copy
import re
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

    def test_base_termination_repr_value_based(self):
        t1 = pybamm.step.VoltageTermination(4.2)
        t2 = pybamm.step.VoltageTermination(4.2)
        assert repr(t1) == repr(t2) == "VoltageTermination(4.2)"
        assert re.search(r"0x[0-9a-f]+", repr(t1)) is None

        t_gt = pybamm.step.VoltageTermination(4.2, operator=">")
        t_lt = pybamm.step.VoltageTermination(4.2, operator="<")
        assert repr(t_gt) == "VoltageTermination(4.2 >)"
        assert repr(t_lt) == "VoltageTermination(4.2 <)"

        assert repr(pybamm.step.CRateTermination(0.01)) == "CRateTermination(0.01)"
        assert (
            repr(pybamm.step.CurrentTermination(0.5, operator=">"))
            == "CurrentTermination(0.5 >)"
        )

    def test_base_termination_hash(self):
        t1 = pybamm.step.VoltageTermination(4.2, operator="<")
        t2 = pybamm.step.VoltageTermination(4.2, operator="<")
        t3 = pybamm.step.VoltageTermination(4.2, operator=">")
        t4 = pybamm.step.VoltageTermination(2.5, operator="<")
        t5 = pybamm.step.CurrentTermination(4.2, operator="<")

        assert hash(t1) == hash(t2)
        assert hash(t1) != hash(t3)
        assert hash(t1) != hash(t4)
        assert hash(t1) != hash(t5)
        assert {t1, t2, t3, t4, t5} == {t1, t3, t4, t5}

    def test_base_termination_eq_strict_type_and_operator(self):
        base = pybamm.step.BaseTermination(1.0)
        current = pybamm.step.CurrentTermination(1.0)
        voltage = pybamm.step.VoltageTermination(1.0)
        assert base != current
        assert current != voltage
        assert pybamm.step.VoltageTermination(
            4.2, operator="<"
        ) != pybamm.step.VoltageTermination(4.2, operator=">")
        assert pybamm.step.VoltageTermination(
            4.2, operator="<"
        ) != pybamm.step.VoltageTermination(4.2)
        assert pybamm.step.VoltageTermination(
            4.2, operator="<"
        ) == pybamm.step.VoltageTermination(4.2, operator="<")

    def test_custom_termination_repr_eq_hash(self):
        def evt(variables):
            return variables["x"] - 1.0

        def evt2(variables):
            return variables["x"] - 2.0

        t1 = pybamm.step.CustomTermination(name="cut-off", event_function=evt)
        t2 = pybamm.step.CustomTermination(name="cut-off", event_function=evt)
        t3 = pybamm.step.CustomTermination(name="other", event_function=evt)
        t4 = pybamm.step.CustomTermination(name="cut-off", event_function=evt2)

        assert repr(t1) == "CustomTermination(cut-off [experiment])"
        assert re.search(r"0x[0-9a-f]+", repr(t1)) is None
        assert t1 == t2
        assert hash(t1) == hash(t2)
        assert t1 != t3
        assert t1 != t4
        assert t1 != pybamm.step.VoltageTermination(1.0)
        assert {t1, t2, t3, t4} == {t1, t3, t4}

        t_copy = copy.deepcopy(t1)
        assert t_copy == t1
        assert hash(t_copy) == hash(t1)
        assert repr(t_copy) == repr(t1)

    def test_deepcopied_termination_is_value_equal(self):
        t = pybamm.step.VoltageTermination(2.5, operator="<")
        t_copy = copy.deepcopy(t)
        assert t_copy is not t
        assert t_copy == t
        assert hash(t_copy) == hash(t)
        assert repr(t_copy) == repr(t)

    def test_unique_steps_independent_of_cycle_count(self):
        # unique_steps must equal template length regardless of n_cycles
        cycle_template = [
            {
                "type": "c-rate",
                "value": 1.0,
                "duration": 3600.0,
                "terminations": [{"type": "voltage", "value": 2.5}],
            },
            {
                "type": "c-rate",
                "value": -0.3,
                "duration": 24000.0,
                "terminations": [{"type": "voltage", "value": 4.2}],
            },
            {
                "type": "voltage",
                "value": 4.2,
                "duration": 86400.0,
                "terminations": [{"type": "c-rate", "value": 0.01}],
            },
        ]
        n_unique = len(cycle_template)

        for n_cycles in (1, 2, 5, 10):
            config = {
                "cycles": [copy.deepcopy(cycle_template) for _ in range(n_cycles)]
            }
            exp = pybamm.Experiment.from_config(config)

            assert len(exp.steps) == n_cycles * n_unique
            assert len(exp.unique_steps) == n_unique, (
                f"unique_steps={len(exp.unique_steps)} for n_cycles={n_cycles}, "
                f"expected {n_unique} (steps must scale with template, not cycles)"
            )
