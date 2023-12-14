#
# Test the experiment step termination classes
#
import pybamm
import unittest


class TestExperimentStepTermination(unittest.TestCase):
    def test_base_termination(self):
        term = pybamm.step.BaseTermination(1)
        self.assertEqual(term.value, 1)

        with self.assertRaises(NotImplementedError):
            term.get_event(None, None)

        self.assertEqual(term, pybamm.step.BaseTermination(1))
        self.assertNotEqual(term, pybamm.step.BaseTermination(2))
        self.assertNotEqual(term, pybamm.step.CurrentTermination(1))
