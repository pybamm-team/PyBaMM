#
# Test the experiment steps
#
import pybamm
import unittest
import numpy as np
from datetime import datetime


class TestExperimentSteps(unittest.TestCase):
    def test_step(self):
        step = pybamm.step.current(1, duration=3600)
        self.assertEqual(step.value, 1)
        self.assertEqual(step.duration, 3600)
        self.assertEqual(step.termination, [])
        self.assertEqual(step.period, None)
        self.assertEqual(step.temperature, None)
        self.assertEqual(step.tags, [])
        self.assertEqual(step.start_time, None)
        self.assertEqual(step.end_time, None)
        self.assertEqual(step.next_start_time, None)

        step = pybamm.step.voltage(
            1,
            duration="1h",
            termination="2.5V",
            period="1 minute",
            temperature=298.15,
            tags="test",
            start_time=datetime(2020, 1, 1, 0, 0, 0),
        )
        self.assertEqual(step.value, 1)
        self.assertEqual(step.duration, 3600)
        self.assertEqual(step.termination, [pybamm.step.VoltageTermination(2.5)])
        self.assertEqual(step.period, 60)
        self.assertEqual(step.temperature, 298.15)
        self.assertEqual(step.tags, ["test"])
        self.assertEqual(step.start_time, datetime(2020, 1, 1, 0, 0, 0))

        step = pybamm.step.current(1, temperature="298K")
        self.assertEqual(step.temperature, 298)

        with self.assertRaisesRegex(ValueError, "temperature units"):
            step = pybamm.step.current(1, temperature="298T")

        with self.assertRaisesRegex(ValueError, "time must be positive"):
            pybamm.step.current(1, duration=0)

    def test_specific_steps(self):
        current = pybamm.step.current(1)
        self.assertIsInstance(current, pybamm.step.Current)
        self.assertEqual(current.value, 1)
        self.assertEqual(str(current), repr(current))
        self.assertEqual(current.duration, 24 * 3600)

        c_rate = pybamm.step.c_rate(1)
        self.assertIsInstance(c_rate, pybamm.step.CRate)
        self.assertEqual(c_rate.value, 1)
        self.assertEqual(c_rate.duration, 3600 * 2)

        voltage = pybamm.step.voltage(1)
        self.assertIsInstance(voltage, pybamm.step.Voltage)
        self.assertEqual(voltage.value, 1)

        rest = pybamm.step.rest()
        self.assertIsInstance(rest, pybamm.step.Current)
        self.assertEqual(rest.value, 0)

        power = pybamm.step.power(1)
        self.assertIsInstance(power, pybamm.step.Power)
        self.assertEqual(power.value, 1)

        resistance = pybamm.step.resistance(1)
        self.assertIsInstance(resistance, pybamm.step.Resistance)
        self.assertEqual(resistance.value, 1)

    def test_step_string(self):
        steps = [
            "Discharge at 1C for 0.5 hours",
            "Discharge at C/20 for 1h (2 minute period)",
            "Charge at 0.5 C for 45 minutes",
            "Discharge at 1 A for 0.5 hours",
            "Charge at 200 mA for 45 minutes",
            "Discharge at 1W for 0.5 hours",
            "Charge at 200mW for 45 minutes",
            "Rest for 10 minutes",
            "Hold at 1V for 20 seconds",
            "Charge at 1 C until 4.1V",
            "Hold at 4.1 V until 50mA",
            "Hold at 3V until C/50",
            "Discharge at C/3 for 2 hours or until 2.5 V",
        ]

        expected_result = [
            {
                "type": "CRate",
                "value": 1.0,
                "duration": 1800.0,
                "termination": [],
            },
            {
                "type": "CRate",
                "value": 0.05,
                "duration": 3600.0,
                "termination": [],
                "period": 120,
            },
            {
                "type": "CRate",
                "value": -0.5,
                "duration": 2700.0,
                "termination": [],
            },
            {
                "value": 1.0,
                "type": "Current",
                "duration": 1800.0,
                "termination": [],
            },
            {
                "value": -0.2,
                "type": "Current",
                "duration": 2700.0,
                "termination": [],
            },
            {
                "value": 1.0,
                "type": "Power",
                "duration": 1800.0,
                "termination": [],
            },
            {
                "value": -0.2,
                "type": "Power",
                "duration": 2700.0,
                "termination": [],
            },
            {
                "value": 0,
                "type": "Current",
                "duration": 600.0,
                "termination": [],
            },
            {
                "value": 1,
                "type": "Voltage",
                "duration": 20.0,
                "termination": [],
            },
            {
                "type": "CRate",
                "value": -1,
                "duration": 7200,
                "termination": [pybamm.step.VoltageTermination(4.1)],
            },
            {
                "value": 4.1,
                "type": "Voltage",
                "duration": 3600 * 24,
                "termination": [pybamm.step.CurrentTermination(0.05)],
            },
            {
                "value": 3,
                "type": "Voltage",
                "duration": 3600 * 24,
                "termination": [pybamm.step.CrateTermination(0.02)],
            },
            {
                "type": "CRate",
                "value": 1 / 3,
                "duration": 7200.0,
                "termination": [pybamm.step.VoltageTermination(2.5)],
            },
        ]

        for step, expected in zip(steps, expected_result):
            actual = pybamm.step.string(step).to_dict()
            for k in expected.keys():
                # useful form for debugging
                self.assertEqual([k, expected[k]], [k, actual[k]])

        with self.assertRaisesRegex(ValueError, "Period cannot be"):
            pybamm.step.string(
                "Discharge at 1C for 1 hour (1 minute period)", period=60
            )

        with self.assertRaisesRegex(ValueError, "Temperature must be"):
            pybamm.step.string("Discharge at 1C for 1 hour at 298.15oC")

    def test_drive_cycle(self):
        # Import drive cycle from file
        drive_cycle = np.array([np.arange(10), np.arange(10)]).T

        # Create steps
        drive_cycle_step = pybamm.step.current(drive_cycle, temperature="-5oC")
        # Check drive cycle operating conditions
        self.assertEqual(drive_cycle_step.duration, 9)
        self.assertEqual(drive_cycle_step.period, 1)
        self.assertEqual(drive_cycle_step.temperature, 273.15 - 5)

        bad_drive_cycle = np.ones((10, 3))
        with self.assertRaisesRegex(ValueError, "Drive cycle must be a 2-column array"):
            pybamm.step.current(bad_drive_cycle)

    def test_drive_cycle_duration(self):
        # Import drive cycle from file
        drive_cycle = np.array([np.arange(10), np.arange(10)]).T

        # Check duration longer than drive cycle data
        # Create steps
        drive_cycle_step = pybamm.step.current(
            drive_cycle, duration=20, temperature="-5oC"
        )
        # Check drive cycle operating conditions
        self.assertEqual(drive_cycle_step.duration, 20)
        self.assertEqual(drive_cycle_step.period, 1)
        self.assertEqual(drive_cycle_step.temperature, 273.15 - 5)

        # Check duration shorter than drive cycle data
        # Create steps
        drive_cycle_step = pybamm.step.current(
            drive_cycle, duration=5, temperature="-5oC"
        )
        # Check drive cycle operating conditions
        self.assertEqual(drive_cycle_step.duration, 5)
        self.assertEqual(drive_cycle_step.period, 1)
        self.assertEqual(drive_cycle_step.temperature, 273.15 - 5)

    def test_bad_strings(self):
        with self.assertRaisesRegex(TypeError, "Input to step.string"):
            pybamm.step.string(1)
        with self.assertRaisesRegex(TypeError, "Input to step.string"):
            pybamm.step.string((1, 2, 3))
        with self.assertRaisesRegex(ValueError, "Operating conditions must"):
            pybamm.step.string("Discharge at 1 A at 2 hours")
        with self.assertRaisesRegex(ValueError, "drive cycles"):
            pybamm.step.string("Run at 1 A for 2 hours")
        with self.assertRaisesRegex(ValueError, "Instruction must be"):
            pybamm.step.string("Play at 1 A for 2 hours")
        with self.assertRaisesRegex(ValueError, "Operating conditions must"):
            pybamm.step.string("Do at 1 A")
        with self.assertRaisesRegex(ValueError, "Instruction"):
            pybamm.step.string("Cell Charge at 1 A for 2 hours")
        with self.assertRaisesRegex(ValueError, "units must be"):
            pybamm.step.string("Discharge at 1 B for 2 hours")
        with self.assertRaisesRegex(ValueError, "time units must be"):
            pybamm.step.string("Discharge at 1 A for 2 years")

    def test_start_times(self):
        # Test start_times
        step = pybamm.step.current(
            1, duration=3600, start_time=datetime(2020, 1, 1, 0, 0, 0)
        )
        self.assertEqual(step.start_time, datetime(2020, 1, 1, 0, 0, 0))

        # Test bad start_times
        with self.assertRaisesRegex(TypeError, "`start_time` should be"):
            pybamm.step.current(1, duration=3600, start_time="bad start_time")

    def test_custom_termination(self):
        def neg_stoich_cutoff(variables):
            return variables["Negative electrode stoichiometry"] - 1

        neg_stoich_termination = pybamm.step.CustomTermination(
            name="Negative stoichiometry cut-off", event_function=neg_stoich_cutoff
        )
        variables = {"Negative electrode stoichiometry": 3}
        event = neg_stoich_termination.get_event(variables, None)
        self.assertEqual(event.name, "Negative stoichiometry cut-off [experiment]")
        self.assertEqual(event.expression, 2)

    def test_drive_cycle_start_time(self):
        # An example where start_time t>0
        t = np.array([[1, 1], [2, 2], [3, 3]])

        with self.assertRaisesRegex(ValueError, "Drive cycle must start at t=0"):
            pybamm.step.current(t)

    def test_base_custom_steps(self):
        with self.assertRaises(NotImplementedError):
            pybamm.step.BaseStepExplicit(None).current_value(None)
        with self.assertRaises(NotImplementedError):
            pybamm.step.BaseStepImplicit(None).get_submodel(None)

    def test_custom_steps(self):
        def custom_step_constant(variables):
            return 1

        custom_constant = pybamm.step.CustomStepExplicit(custom_step_constant)

        self.assertEqual(custom_constant.current_value_function({}), 1)

        def custom_step_voltage(variables):
            return variables["Voltage [V]"] - 4.1

        custom_step_alg = pybamm.step.CustomStepImplicit(custom_step_voltage)

        self.assertEqual(custom_step_alg.control, "algebraic")
        self.assertAlmostEqual(
            custom_step_alg.current_rhs_function({"Voltage [V]": 4.2}), 0.1
        )

        custom_step_diff = pybamm.step.CustomStepImplicit(
            custom_step_voltage, control="differential"
        )
        self.assertEqual(custom_step_diff.control, "differential")

        with self.assertRaisesRegex(ValueError, "control must be"):
            pybamm.step.CustomStepImplicit(custom_step_voltage, control="bla")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
