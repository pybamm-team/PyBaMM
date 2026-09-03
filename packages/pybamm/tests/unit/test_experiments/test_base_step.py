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

    # One cycle, plus the wrap point that repeats the first sample: the step time is
    # wrapped into this window rather than the cycle being tiled out to the duration
    np.testing.assert_array_equal(step.value.x[0], np.array([0.0, 2.0, 4.0]))
    np.testing.assert_array_equal(step.value.y, np.array([0.0, 1.0, 0.0]))

    # The drive cycle repeats every 4 s, for as long as the step runs
    def current_at(t):
        return step.value.evaluate(t=t, inputs={"start time": 0.0})

    for t in (0.0, 1.0, 2.0, 3.0):
        assert current_at(t) == current_at(t + 4.0) == current_at(t + 8.0)
    assert current_at(1.0) == 0.5


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


def test_symbolic_duration_is_resolved_at_solve_time():
    step = pybamm.step.c_rate(1, duration=pybamm.InputParameter("C-rate duration"))

    assert isinstance(step.duration, pybamm.InputParameter)
    assert not step.uses_default_duration
    assert step.duration_seconds({"C-rate duration": 1200}) == 1200.0
    # A numeric duration is returned unchanged, with or without inputs
    assert pybamm.step.c_rate(1, duration=600).duration_seconds() == 600


def test_symbolic_duration_must_be_positive_when_resolved():
    step = pybamm.step.c_rate(1, duration=pybamm.InputParameter("d"))

    with pytest.raises(ValueError, match="duration must be positive"):
        step.duration_seconds({"d": -1})


def test_symbolic_duration_needs_its_input():
    step = pybamm.step.c_rate(1, duration=pybamm.InputParameter("d"))

    with pytest.raises(KeyError, match="Input parameter 'd' not found"):
        step.duration_seconds()


def test_drive_cycle_repeats_independently_of_its_duration():
    drive_cycle = np.array([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5]])
    symbolic = pybamm.step.current(drive_cycle, duration=pybamm.InputParameter("d"))
    numeric = pybamm.step.current(drive_cycle, duration=10)

    # The interpolant wraps the step time, so it does not depend on the duration
    assert symbolic.value == numeric.value
    assert symbolic.duration_seconds({"d": 10}) == 10
    # Sample times are repeated up to the resolved final time, one cycle being 3 s
    np.testing.assert_array_equal(
        symbolic._drive_cycle_time_points(8),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )


def test_symbolic_duration_and_termination_recorded_by_id():
    termination = pybamm.CoupledVariable("Voltage [V]") < pybamm.InputParameter("Vmin")
    duration = pybamm.InputParameter("d")
    step = pybamm.step.c_rate(1, duration=duration, termination=termination)

    # A Symbol has no truth value and no stable repr, so its id stands in for it
    assert f"duration={duration.id}" in repr(step)
    assert f"termination={termination.id}" in step.basic_repr()
    # Steps differing only in duration still share a model
    assert (
        step.basic_repr()
        == pybamm.step.c_rate(1, duration=1800, termination=termination).basic_repr()
    )
