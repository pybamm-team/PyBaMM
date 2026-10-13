import numpy as np
import pytest

import pybamm


class DummySolver:
    supports_interp = True
    store_first_last = False

    @staticmethod
    def process_t_interp(t_interp):
        return t_interp


@pytest.mark.parametrize(
    "test_string, unit_string",
    [
        ("123e-1 W", "W"),
        ("123K", "K"),
        ("1A", "A"),
        ("2.0 mV", "mV"),
        ("0.5 Ohm", "Ohm"),
        ("1e0hours", "hours"),
    ],
)
def test_read_units(test_string, unit_string):
    assert unit_string == pybamm.experiment.step.base_step.get_unit_from(test_string)


def test_drive_cycle_default_time_vector_uses_smallest_sample_spacing():
    drive_cycle = np.array([[0.0, 0.0], [2.0, 1.0], [5.0, -1.0]])
    step = pybamm.step.current(drive_cycle, duration=4)

    time_vector = step.default_time_vector(DummySolver(), tf=4)

    np.testing.assert_array_equal(time_vector, np.array([0.0, 2.0, 4.0]))


def test_drive_cycle_setup_timestepping_truncates_and_appends_final_time():
    drive_cycle = np.array([[0.0, 0.0], [2.0, 1.0], [5.0, -1.0]])
    step = pybamm.step.current(drive_cycle, duration=4)

    t_eval, t_interp = step.setup_timestepping(DummySolver(), tf=4)

    np.testing.assert_array_equal(t_eval, np.array([0.0, 2.0, 4.0]))
    assert t_interp is None


def test_base_step_validates_direction_tags_and_start_time():
    with pytest.raises(ValueError, match="Invalid direction"):
        pybamm.step.current(1, duration=10, direction="sideways")

    step = pybamm.step.current(1, duration=10, tags="tag")
    assert step.tags == ["tag"]

    with pytest.raises(TypeError, match=r"datetime\.datetime"):
        pybamm.step.current(1, duration=10, start_time=0)


def test_drive_cycle_validation_and_looping():
    with pytest.raises(ValueError, match="2-column array"):
        pybamm.step.current(np.array([0.0, 1.0]), duration=1)

    with pytest.raises(ValueError, match="start at t=0"):
        pybamm.step.current(np.array([[1.0, 0.0], [2.0, 1.0]]), duration=2)

    step = pybamm.step.current(np.array([[0.0, 0.0], [2.0, 1.0]]), duration=5)

    np.testing.assert_array_equal(
        step.value.x[0], np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    )
    np.testing.assert_array_equal(
        step.value.y, np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    )


def test_default_time_vector_uses_drive_cycle_period_only_when_supported():
    step = pybamm.step.current(
        np.array([[0.0, 0.0], [2.0, 1.0], [5.0, -1.0]]), duration=5
    )

    class SolverWithoutInterp:
        supports_interp = False
        store_first_last = False

    time_vector = step.default_time_vector(SolverWithoutInterp(), tf=5)

    np.testing.assert_array_equal(time_vector, np.array([0.0, 5.0]))


def test_value_based_charge_or_discharge_with_input_parameter():
    step = pybamm.step.current(pybamm.InputParameter("I_app"), termination="> 2A")

    assert step.direction is None
    assert step.termination[0] == pybamm.step.CurrentTermination(2.0, operator=">")


@pytest.mark.parametrize(
    "helper,input_value,match",
    [
        (
            pybamm.experiment.step.base_step._convert_time_to_seconds,
            0,
            "time must be positive",
        ),
        (
            pybamm.experiment.step.base_step._convert_time_to_seconds,
            "1 day",
            "time units must be 'seconds', 'minutes' or 'hours'",
        ),
        (
            pybamm.experiment.step.base_step._convert_temperature_to_kelvin,
            "20 F",
            "temperature units must be 'K' or 'oC'",
        ),
        (
            pybamm.experiment.step.base_step._convert_electric,
            "1 B",
            "units must be 'A', 'V', 'W', 'Ohm', or 'C'",
        ),
    ],
)
def test_base_step_conversion_helpers_raise_for_invalid_inputs(
    helper, input_value, match
):
    with pytest.raises(ValueError, match=match):
        helper(input_value)


def test_parse_termination_requires_operator_for_input_parameter():
    with pytest.raises(ValueError, match="Termination must include an operator"):
        pybamm.experiment.step.base_step._parse_termination(
            "2A", pybamm.InputParameter("I_app")
        )


def _t_dependent_current(t):
    """A Python-function current that genuinely depends on ``t`` so that
    ``BaseStep`` cannot fold ``value(t)`` into a scalar at construction.
    Module-scope so steps stay picklable (used elsewhere in this file)."""
    return -1.0 + 0.1 * np.sin(2 * np.pi * t)


def test_check_input_params_ignores_internal_start_time():
    """Regression for #5018. ``BaseStep`` injects
    ``InputParameter("start time")`` into ``self.value`` whenever the
    step is built from a Python function (or a drive cycle). That
    placeholder is an implementation detail — it is not a user-supplied
    ``InputParameter`` — so ``_check_input_params`` must return False
    for a value tree that only depends on it."""
    step = pybamm.step.current(_t_dependent_current, period="0.1 seconds")
    # Sanity check that the internal "start time" placeholder is in the
    # symbol tree, so the test exercises the intended code path.
    leaves = step.value.post_order(filter=lambda n: len(n.children) == 0)
    names = {leaf.name for leaf in leaves if isinstance(leaf, pybamm.InputParameter)}
    assert names == {"start time"}
    assert pybamm.experiment.step.base_step._check_input_params(step.value) is False


def test_check_input_params_still_detects_user_input_parameter():
    """Counterpart to the test above: ``_check_input_params`` must keep
    returning True for value trees that mix the internal ``"start time"``
    placeholder with a genuine user ``InputParameter``. Forces the new
    filter to be ``name != "start time"`` rather than a blanket skip."""
    I_app = pybamm.InputParameter("I_app")
    # Manually build a tree that contains both leaves to make the
    # mixing explicit.
    value = I_app + (pybamm.t - pybamm.InputParameter("start time"))
    assert pybamm.experiment.step.base_step._check_input_params(value) is True


def test_python_function_step_accepts_string_termination_without_operator():
    """End-to-end regression for #5018: ``pybamm.step.current(callable)``
    used to raise ``ValueError: Termination must include an operator
    when using InputParameter.`` whenever the callable depended on
    ``t``, because ``_check_input_params`` saw the internal
    ``InputParameter("start time")`` and assumed the user had passed
    one. After the fix, a plain ``"4.3 V"`` (and other operator-free
    string terminations) must be accepted again."""
    step = pybamm.step.current(
        _t_dependent_current,
        period="0.05 seconds",
        termination=["4.3 V"],
    )
    assert len(step.termination) == 1
    assert step.termination[0] == pybamm.step.VoltageTermination(4.3)


def test_python_function_step_accepts_mixed_terminations():
    """Pins the exact shape of the OP's failing experiment in #5018:
    a ``CustomTermination`` object together with a plain string
    termination, around a ``t``-dependent callable current."""

    def soc_cutoff(variables):
        return variables["Discharge capacity [A.h]"] + 4.5

    soc_termination = pybamm.step.CustomTermination(
        name="SOC 90%", event_function=soc_cutoff
    )
    step = pybamm.step.current(
        _t_dependent_current,
        period="0.05 seconds",
        termination=[soc_termination, "4.3 V"],
    )
    assert len(step.termination) == 2
    assert step.termination[1] == pybamm.step.VoltageTermination(4.3)


def test_python_function_step_infers_direction_from_value_at_zero():
    """After #5018's fix, ``_check_input_params`` no longer flags
    Python-function values as "depends on InputParameter", so
    ``value_based_charge_or_discharge`` falls through to evaluating the
    symbolic value at ``t=0`` (with the internal ``"start time"``
    placeholder set to 0). The sign of the result then drives
    direction inference, restoring the pre-#4826 behaviour for
    callables. ``_t_dependent_current(0) = -1.0`` ⇒ charge."""
    step = pybamm.step.current(_t_dependent_current, period="0.05 seconds")
    assert step.direction == "charge"


def test_user_input_parameter_step_still_requires_operator():
    """Belt-and-braces non-regression: the ``InputParameter`` →
    "must include an operator" guard introduced by #4826 must keep
    firing when the user genuinely passes one. Without this we would
    silently weaken the check."""
    with pytest.raises(ValueError, match="Termination must include an operator"):
        pybamm.step.current(pybamm.InputParameter("I_app"), termination="2.5 V")
