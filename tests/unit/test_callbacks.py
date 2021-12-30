#
# Tests the citations class.
#
import pybamm
import unittest
import os
from pybamm import callbacks


class DummyCallback(callbacks.Callback):
    def __init__(self, logs, name):
        self.name = name
        self.logs = logs

    def on_simulation_end(self, *args):
        with open(self.logs, "w") as f:
            print(self.name, file=f)


class DummySimulation:
    solution = pybamm.Solution(
        [0], [1, 2, 3], pybamm.BaseModel(), {}, termination="final time"
    )


class TestCallbacks(unittest.TestCase):
    def setUp(self):
        self.sim = DummySimulation()

    def tearDown(self):
        del self.sim

    def test_setup_callbacks(self):
        callbacks = pybamm.callbacks.setup_callbacks([1, 2, 3])
        self.assertIsInstance(callbacks, pybamm.callbacks.CallbackList)
        self.assertEqual(callbacks.callbacks, [1, 2, 3])

        callbacks = pybamm.callbacks.setup_callbacks(1)
        self.assertIsInstance(callbacks, pybamm.callbacks.CallbackList)
        self.assertEqual(callbacks.callbacks, [1])

    def test_logging_callback(self):
        callback = pybamm.callbacks.LoggingCallback("test_callback.log")
        self.assertEqual(callback.logs, "test_callback.log")
        callback.on_simulation_end(self.sim)
        with open("test_callback.log", "r") as f:
            self.assertEqual(f.read(), "final time\n")
        os.remove("test_callback.log")

    def test_callback_list(self):
        "Tests multiple callbacks in a list"
        # Should work with empty callback list
        callbacks = pybamm.callbacks.CallbackList([])
        callbacks.on_simulation_end()

        # Should work with multiple callbacks
        callback = pybamm.callbacks.CallbackList(
            [
                DummyCallback("test_callback.log", "first"),
                DummyCallback("test_callback_2.log", "second"),
            ]
        )
        callback.on_simulation_end(self.sim)
        with open("test_callback.log", "r") as f:
            self.assertEqual(f.read(), "first\n")
        with open("test_callback_2.log", "r") as f:
            self.assertEqual(f.read(), "second\n")

        os.remove("test_callback.log")
        os.remove("test_callback_2.log")


if __name__ == "__main__":
    print("Add -v for more debug output")
    import sys

    if "-v" in sys.argv:
        debug = True
    pybamm.settings.debug_mode = True
    unittest.main()
