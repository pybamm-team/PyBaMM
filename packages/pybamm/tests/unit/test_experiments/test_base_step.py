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


def _crate_one(t):
    """A C-rate callable that ignores time. Defined at module scope so
    that the regression tests below carry a real ``callable(value) is True``
    rather than relying on lambdas that ``pickle`` cannot round-trip."""
    return 1.0


def test_crate_default_duration_with_callable_falls_back_to_24h():
    """Guards #4926: ``CRate(callable)`` used to raise
    ``TypeError: bad operand type for abs(): 'function'`` from
    ``CRate._default_timespan`` evaluating ``1 / abs(value) * 3600 * 2``
    on the function object instead of its return value.

    The fix delegates to the base class's 24-hour default whenever the
    value is callable, matching how every other ``step.*`` subclass
    (``Current``, ``Voltage``, ``Power``, ``Resistance``) behaves for
    callables. ``CRate(scalar)`` keeps the C-rate-derived bound."""
    step = pybamm.step.CRate(_crate_one)
    assert step.duration == 24 * 3600
    assert step.uses_default_duration is True


def test_crate_default_duration_with_scalar_unchanged():
    """Non-regression for the existing fast path: a numeric C-rate
    still resolves to ``1 / |C| * 3600 * 2`` seconds."""
    step = pybamm.step.CRate(0.5)
    assert step.duration == pytest.approx(1 / 0.5 * 3600 * 2)
    step = pybamm.step.CRate(-2.0)
    assert step.duration == pytest.approx(1 / 2.0 * 3600 * 2)


def test_all_step_types_accept_callable_for_default_duration():
    """Every explicit/implicit step subclass must accept a callable
    without crashing in ``default_duration``. Before #4926's fix only
    ``CRate`` was broken; this test pins all five to the same contract."""
    for step_cls in (
        pybamm.step.Current,
        pybamm.step.CRate,
        pybamm.step.Voltage,
        pybamm.step.Power,
        pybamm.step.Resistance,
    ):
        step = step_cls(_crate_one)
        assert step.duration == 24 * 3600
        # ``BaseStep.__init__`` materialises ``value`` by calling the
        # function at ``t=0``, so ``step.value`` is the scalar result
        # rather than the function object; the contract under test is
        # that construction succeeds with the 24-hour fallback duration.
        assert step.is_python_function is True
