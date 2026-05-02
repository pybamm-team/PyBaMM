"""
Regression tests for historical parameter handling bug fixes.

These tests guard against the reintroduction of bugs that were fixed but
did not have regression tests added at the time of the fix.
"""

import pytest

import pybamm


class TestParameterValuesFixes:
    """Guards for parameter values handling bug fixes."""

    def test_depreciated_parameter_raises_error(self):
        """
        Guards against: 388d1366f - Bug Fix (parameter deprecation message)

        The bug was an incorrect condition "name in self.keys() == ..." which
        always evaluated to False, meaning the deprecation check never triggered.
        The fix changed it to "name == ..." so the error is actually raised.
        """
        param = pybamm.ParameterValues("Chen2020")

        with pytest.raises(ValueError):
            param.update({"1 + dlnf/dlnc": 1.5})
