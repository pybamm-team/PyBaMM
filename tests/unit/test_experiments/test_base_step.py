import numpy as np
import pytest

import pybamm


class DummySolver:
    supports_interp = True

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
