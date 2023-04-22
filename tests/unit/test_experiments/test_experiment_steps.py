#
# Test the experiment steps
#
import pybamm
import unittest


class TestExperimentSteps(unittest.TestCase):
    def test_step(self):
        step = pybamm.experiment._Step("current", 1, duration=3600)
        self.assertEqual(step.typ, "current")
        self.assertEqual(step.value, 1)
        self.assertEqual(step.duration, 3600)
        self.assertEqual(step.termination, None)
        self.assertEqual(step.period, None)
        self.assertEqual(step.temperature, None)
        self.assertEqual(step.tags, [])

        step = pybamm.experiment._Step(
            "voltage",
            1,
            duration="1h",
            termination="2.5V",
            period="1 minute",
            temperature=298.15,
            tags="test",
        )
        self.assertEqual(step.typ, "voltage")
        self.assertEqual(step.value, 1)
        self.assertEqual(step.duration, 3600)
        self.assertEqual(step.termination, {"type": "voltage", "value": 2.5})
        self.assertEqual(step.period, 60)
        self.assertEqual(step.temperature, 298.15)
        self.assertEqual(step.tags, ["test"])

    def test_specific_steps(self):
        current = pybamm.experiment.current(1)
        self.assertIsInstance(current, pybamm.experiment._Step)
        self.assertEqual(current.typ, "current")
        self.assertEqual(current.value, 1)

        voltage = pybamm.experiment.voltage(1)
        self.assertIsInstance(voltage, pybamm.experiment._Step)
        self.assertEqual(voltage.typ, "voltage")
        self.assertEqual(voltage.value, 1)

        rest = pybamm.experiment.rest()
        self.assertIsInstance(rest, pybamm.experiment._Step)
        self.assertEqual(rest.typ, "current")
        self.assertEqual(rest.value, 0)

        cccv = pybamm.experiment.cccv_ode(1, 2)
        self.assertIsInstance(cccv, pybamm.experiment._Step)
        self.assertEqual(cccv.typ, "cccv_ode")
        self.assertEqual(cccv.value, (1, 2))

    def test_step_string(self):
        steps = [
            "Discharge at 1C for 0.5 hours",
            "Discharge at C/20 for 0.5 hours",
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
                "type": "C-rate",
                "value": 1.0,
                "duration": 1800.0,
                "termination": None,
            },
            {
                "type": "C-rate",
                "value": 0.05,
                "duration": 1800.0,
                "termination": None,
            },
            {
                "type": "C-rate",
                "value": -0.5,
                "duration": 2700.0,
                "termination": None,
            },
            {
                "value": 1.0,
                "type": "current",
                "duration": 1800.0,
                "termination": None,
            },
            {
                "value": -0.2,
                "type": "current",
                "duration": 2700.0,
                "termination": None,
            },
            {
                "value": 1.0,
                "type": "power",
                "duration": 1800.0,
                "termination": None,
            },
            {
                "value": -0.2,
                "type": "power",
                "duration": 2700.0,
                "termination": None,
            },
            {
                "value": 0,
                "type": "current",
                "duration": 600.0,
                "termination": None,
            },
            {
                "value": 1,
                "type": "voltage",
                "duration": 20.0,
                "termination": None,
            },
            {
                "type": "C-rate",
                "value": -1,
                "duration": None,
                "termination": {"value": 4.1, "type": "voltage"},
            },
            {
                "value": 4.1,
                "type": "voltage",
                "duration": None,
                "termination": {"value": 0.05, "type": "current"},
            },
            {
                "value": 3,
                "type": "voltage",
                "duration": None,
                "termination": {"value": 0.02, "type": "C-rate"},
            },
            {
                "type": "C-rate",
                "value": 1 / 3,
                "duration": 7200.0,
                "termination": {"value": 2.5, "type": "voltage"},
            },
        ]

        for step, expected in zip(steps, expected_result):
            print(step)
            actual = pybamm.experiment.string(step).to_dict()
            for k in expected.keys():
                # useful form for debugging
                self.assertEqual([k, expected[k]], [k, actual[k]])

    # def test_drive_cycle(self):
    #     # Import drive cycle from file
    #     drive_cycle = pd.read_csv(
    #         pybamm.get_parameters_filepath(
    #             os.path.join("input", "drive_cycles", "US06.csv")
    #         ),
    #         comment="#",
    #         header=None,
    #     ).to_numpy()

    #     # Calculation for operating conditions of drive cycle
    #     time_0 = drive_cycle[:, 0][-1]
    #     period_0 = np.min(np.diff(drive_cycle[:, 0]))
    #     drive_cycle_1 = experiment.extend_drive_cycle(drive_cycle, end_time=300)
    #     time_1 = drive_cycle_1[:, 0][-1]
    #     period_1 = np.min(np.diff(drive_cycle_1[:, 0]))
    #     drive_cycle_2 = experiment.extend_drive_cycle(drive_cycle, end_time=1800)
    #     time_2 = drive_cycle_2[:, 0][-1]
    #     period_2 = np.min(np.diff(drive_cycle_2[:, 0]))
    #     # Check drive cycle operating conditions
    #     np.testing.assert_array_equal(
    #         experiment.operating_conditions[-3]["dc_data"], drive_cycle
    #     )
    #     self.assertEqual(experiment.operating_conditions[-3]["type"], "current")
    #     self.assertEqual(experiment.operating_conditions[-3]["time"], time_0)
    #     self.assertEqual(experiment.operating_conditions[-3]["period"], period_0)
    #     self.assertEqual(experiment.operating_conditions[-3]["temperature"], -5)
    #     self.assertEqual(experiment.operating_conditions[-3]["tags"], None)
    #     np.testing.assert_array_equal(
    #         experiment.operating_conditions[-2]["dc_data"], drive_cycle_1
    #     )
    #     self.assertEqual(experiment.operating_conditions[-2]["type"], "voltage")
    #     self.assertEqual(experiment.operating_conditions[-2]["time"], time_1)
    #     self.assertEqual(experiment.operating_conditions[-2]["period"], period_1)
    #     self.assertEqual(experiment.operating_conditions[-2]["tags"], None)
    #     np.testing.assert_array_equal(
    #         experiment.operating_conditions[-1]["dc_data"], drive_cycle_2
    #     )
    #     self.assertEqual(experiment.operating_conditions[-1]["type"], "power")
    #     self.assertEqual(experiment.operating_conditions[-1]["time"], time_2)
    #     self.assertEqual(experiment.operating_conditions[-1]["period"], period_2)
    #     self.assertEqual(experiment.operating_conditions[-1]["tags"], None)
    #     self.assertEqual(experiment.period, 20)

    def test_bad_strings(self):
        with self.assertRaisesRegex(TypeError, "Input to experiment.string"):
            pybamm.experiment.string(1)
        with self.assertRaisesRegex(TypeError, "Input to experiment.string"):
            pybamm.experiment.string((1, 2, 3))
        with self.assertRaisesRegex(ValueError, "Operating conditions must"):
            pybamm.experiment.string("Discharge at 1 A at 2 hours")
        with self.assertRaisesRegex(ValueError, "Instruction must be"):
            pybamm.experiment.string("Run at 1 A for 2 hours")
        with self.assertRaisesRegex(ValueError, "Instruction must be"):
            pybamm.experiment.string("Run at at 1 A for 2 hours")
        with self.assertRaisesRegex(ValueError, "Instruction must be"):
            pybamm.experiment.string("Play at 1 A for 2 hours")
        with self.assertRaisesRegex(ValueError, "Operating conditions must"):
            pybamm.experiment.string("Do at 1 A")
        with self.assertRaisesRegex(ValueError, "Instruction"):
            pybamm.experiment.string("Cell Charge at 1 A for 2 hours")
        with self.assertRaisesRegex(ValueError, "units must be"):
            pybamm.experiment.string("Discharge at 1 B for 2 hours")
        with self.assertRaisesRegex(ValueError, "time units must be"):
            pybamm.experiment.string("Discharge at 1 A for 2 years")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
