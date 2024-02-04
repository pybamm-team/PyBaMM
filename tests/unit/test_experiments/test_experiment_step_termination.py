#
# Test the experiment step termination classes
#

import pybamm
import pytest


class TestExperimentStepTermination:
    def test_base_termination(self):
        term = pybamm.step.BaseTermination(1)
        assert term.value == 1
        with pytest.raises(NotImplementedError):
            term.get_event(None, None)
        assert term == pybamm.step.BaseTermination(1)
        assert term != pybamm.step.BaseTermination(2)
        assert term != pybamm.step.CurrentTermination(1)
