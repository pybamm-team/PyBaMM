#
# Test the base experiment class
#

from datetime import datetime
import pybamm
import pytest


class TestExperiment:
    def test_cycle_unpacking(self):
        experiment = pybamm.Experiment(
            [
                ("Discharge at C/20 for 0.5 hours", "Charge at C/5 for 45 minutes"),
                ("Discharge at C/20 for 0.5 hours"),
                "Charge at C/5 for 45 minutes",
            ]
        )
        assert [step.to_dict() for step in experiment.steps] == [
            {
                "value": 0.05,
                "type": "CRate",
                "duration": 1800.0,
                "period": None,
                "temperature": None,
                "description": "Discharge at C/20 for 0.5 hours",
                "termination": [],
                "tags": [],
                "start_time": None,
            },
            {
                "value": -0.2,
                "type": "CRate",
                "duration": 2700.0,
                "period": None,
                "temperature": None,
                "description": "Charge at C/5 for 45 minutes",
                "termination": [],
                "tags": [],
                "start_time": None,
            },
            {
                "value": 0.05,
                "type": "CRate",
                "duration": 1800.0,
                "period": None,
                "temperature": None,
                "description": "Discharge at C/20 for 0.5 hours",
                "termination": [],
                "tags": [],
                "start_time": None,
            },
            {
                "value": -0.2,
                "type": "CRate",
                "duration": 2700.0,
                "period": None,
                "temperature": None,
                "description": "Charge at C/5 for 45 minutes",
                "termination": [],
                "tags": [],
                "start_time": None,
            },
        ]
        assert experiment.cycle_lengths == [2, 1, 1]

    def test_invalid_step_type(self):
        unprocessed = {1.0}
        period = 1
        temperature = 300.0
        with pytest.raises(
            TypeError, match="Operating conditions must be a Step object or string."
        ):
            pybamm.Experiment.process_steps(unprocessed, period, temperature)

    def test_str_repr(self):
        conds = ["Discharge at 1 C for 20 seconds", "Charge at 0.5 W for 10 minutes"]
        experiment = pybamm.Experiment(conds)
        assert (
            str(experiment)
            == "[('Discharge at 1 C for 20 seconds',)"
            + ", ('Charge at 0.5 W for 10 minutes',)]"
        )
        assert (
            repr(experiment)
            == "pybamm.Experiment([('Discharge at 1 C for 20 seconds',)"
            + ", ('Charge at 0.5 W for 10 minutes',)])"
        )

    def test_bad_strings(self):
        with pytest.raises(
            TypeError, match="Operating conditions must be a Step object or string."
        ):
            pybamm.Experiment([1, 2, 3])
        with pytest.raises(
            TypeError, match="Operating conditions must be a Step object or string."
        ):
            pybamm.Experiment([(1, 2, 3)])

    def test_termination(self):
        experiment = pybamm.Experiment(["Discharge at 1 C for 20 seconds"])
        assert experiment.termination == {}

        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination=["80.7% capacity"]
        )
        assert experiment.termination == {"capacity": (80.7, "%")}

        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination=["80.7 % capacity"]
        )
        assert experiment.termination == {"capacity": (80.7, "%")}

        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination=["4.1Ah capacity"]
        )
        assert experiment.termination == {"capacity": (4.1, "Ah")}

        experiment = pybamm.Experiment(
            ["Discharge at 1 C for 20 seconds"], termination=["4.1 A.h capacity"]
        )
        assert experiment.termination == {"capacity": (4.1, "Ah")}

        with pytest.raises(ValueError, match="Only capacity"):
            experiment = pybamm.Experiment(
                ["Discharge at 1 C for 20 seconds"], termination="bla bla capacity bla"
            )
        with pytest.raises(ValueError, match="Only capacity"):
            experiment = pybamm.Experiment(
                ["Discharge at 1 C for 20 seconds"], termination="4 A.h something else"
            )
        with pytest.raises(ValueError, match="Capacity termination"):
            experiment = pybamm.Experiment(
                ["Discharge at 1 C for 20 seconds"], termination="1 capacity"
            )

    def test_search_tag(self):
        s = pybamm.step.string
        experiment = pybamm.Experiment(
            [
                (s("Discharge at 1C for 0.5 hours", tags=["tag1"]),),
                s("Discharge at C/20 for 0.5 hours", tags=["tag2", "tag3"]),
                (
                    s("Charge at 0.5 C for 45 minutes", tags=["tag2"]),
                    s("Discharge at 1 A for 0.5 hours", tags=["tag3"]),
                ),
                s("Charge at 200 mA for 45 minutes", tags=["tag5"]),
                (
                    s("Discharge at 1W for 0.5 hours", tags=["tag4"]),
                    s("Charge at 200mW for 45 minutes", tags=["tag4"]),
                ),
                s("Rest for 10 minutes", tags=["tag1", "tag3", "tag4"]),
            ]
        )

        assert experiment.search_tag("tag1") == [0, 5]
        assert experiment.search_tag("tag2") == [1, 2]
        assert experiment.search_tag("tag3") == [1, 2, 5]
        assert experiment.search_tag("tag4") == [4, 5]
        assert experiment.search_tag("tag5") == [3]
        assert experiment.search_tag("no_tag") == []

    def test_no_initial_start_time(self):
        s = pybamm.step.string
        with pytest.raises(ValueError, match="first step must have a `start_time`"):
            pybamm.Experiment(
                [
                    s("Rest for 1 hour"),
                    s("Rest for 1 hour", start_time=datetime(2023, 1, 1, 8, 0)),
                ]
            )

    def test_set_next_start_time(self):
        raw_steps = [
            pybamm.step.Current(
                1, duration=3600, start_time=datetime(2023, 1, 1, 8, 0)
            ),
            pybamm.step.Voltage(2.5, duration=3600, start_time=None),
            pybamm.step.Current(
                1, duration=3600, start_time=datetime(2023, 1, 1, 12, 0)
            ),
            pybamm.step.Current(1, duration=3600, start_time=None),
            pybamm.step.Voltage(2.5, duration=3600, start_time=None),
            pybamm.step.Current(
                1, duration=3600, start_time=datetime(2023, 1, 1, 15, 0)
            ),
        ]
        experiment = pybamm.Experiment(raw_steps)
        processed_steps = experiment._set_next_start_time(raw_steps)

        expected_next = [
            None,
            datetime(2023, 1, 1, 12, 0),
            None,
            None,
            datetime(2023, 1, 1, 15, 0),
            None,
        ]

        expected_end = [
            datetime(2023, 1, 1, 12, 0),
            datetime(2023, 1, 1, 12, 0),
            datetime(2023, 1, 1, 15, 0),
            datetime(2023, 1, 1, 15, 0),
            datetime(2023, 1, 1, 15, 0),
            None,
        ]

        # Test method directly
        for next, end, steps in zip(expected_next, expected_end, processed_steps):
            # useful form for debugging
            assert steps.next_start_time == next
            assert steps.end_time == end

        # TODO: once #3176 is completed, the test should pass for
        # operating_conditions_steps (or equivalent) as well
