#
# Tests the citations class.
#
from tests import TestCase
import pybamm
import unittest
import os
from pybamm import callbacks


class DummyCallback(callbacks.Callback):
    def __init__(self, logs, name):
        self.name = name
        self.logs = logs

    def on_experiment_end(self, logs):
        with open(self.logs, "w") as f:
            print(self.name, file=f)


class TestCallbacks(TestCase):
    def tearDown(self):
        # Remove any test log files that were created, even if the test fails
        for logfile in ["test_callback.log", "test_callback_2.log"]:
            if os.path.exists(logfile):
                try:
                    os.remove(logfile)
                except PermissionError:
                    # Just skip this if it doesn't work (Windows doesn't allow)
                    pass

    def test_setup_callbacks(self):
        # No callbacks, LoggingCallback should be added
        callbacks = pybamm.callbacks.setup_callbacks(None)
        self.assertIsInstance(callbacks, pybamm.callbacks.CallbackList)
        self.assertEqual(len(callbacks), 1)
        self.assertIsInstance(callbacks[0], pybamm.callbacks.LoggingCallback)

        # Single object, transformed to list
        callbacks = pybamm.callbacks.setup_callbacks(1)
        self.assertIsInstance(callbacks, pybamm.callbacks.CallbackList)
        self.assertEqual(len(callbacks), 2)
        self.assertEqual(callbacks.callbacks[0], 1)
        self.assertIsInstance(callbacks[-1], pybamm.callbacks.LoggingCallback)

        # List
        callbacks = pybamm.callbacks.setup_callbacks([1, 2, 3])
        self.assertIsInstance(callbacks, pybamm.callbacks.CallbackList)
        self.assertEqual(callbacks.callbacks[:3], [1, 2, 3])
        self.assertIsInstance(callbacks[-1], pybamm.callbacks.LoggingCallback)

    def test_callback_list(self):
        "Tests multiple callbacks in a list"
        # Should work with empty callback list (does nothiing)
        callbacks = pybamm.callbacks.CallbackList([])
        callbacks.on_experiment_end(None)

        # Should work with multiple callbacks
        callback = pybamm.callbacks.CallbackList(
            [
                DummyCallback("test_callback.log", "first"),
                DummyCallback("test_callback_2.log", "second"),
            ]
        )
        callback.on_experiment_end(None)
        with open("test_callback.log") as f:
            self.assertEqual(f.read(), "first\n")
        with open("test_callback_2.log") as f:
            self.assertEqual(f.read(), "second\n")

    def test_logging_callback(self):
        # No argument, should use pybamm's logger
        callback = pybamm.callbacks.LoggingCallback()
        self.assertEqual(callback.logger, pybamm.logger)

        pybamm.set_logging_level("NOTICE")
        callback = pybamm.callbacks.LoggingCallback("test_callback.log")
        self.assertEqual(callback.logfile, "test_callback.log")

        logs = {
            "cycle number": (5, 12),
            "step number": (1, 4),
            "elapsed time": 0.45,
            "step operating conditions": "Charge",
            "termination": "event",
        }
        callback.on_experiment_start(logs)
        with open("test_callback.log") as f:
            self.assertEqual(f.read(), "")

        callback.on_cycle_start(logs)
        with open("test_callback.log") as f:
            self.assertIn("Cycle 5/12", f.read())

        callback.on_step_start(logs)
        with open("test_callback.log") as f:
            self.assertIn("Cycle 5/12, step 1/4", f.read())

        callback.on_experiment_infeasible(logs)
        with open("test_callback.log") as f:
            self.assertIn("Experiment is infeasible: 'event'", f.read())

        callback.on_experiment_end(logs)
        with open("test_callback.log") as f:
            self.assertIn("took 0.45", f.read())

        # Calling start again should clear the log
        callback.on_experiment_start(logs)
        with open("test_callback.log") as f:
            self.assertEqual(f.read(), "")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
