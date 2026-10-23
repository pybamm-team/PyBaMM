import numpy as np
import pytest

import pybamm


class TestProcessedVariableTimeIntegral:
    @staticmethod
    def _integral():
        return pybamm.ProcessedVariableTimeIntegral(
            method="continuous",
            sum_node=pybamm.Scalar(0),
            initial_condition=0.0,
            discrete_times=None,
        )

    def test_boundary_sensitivities_add_to_the_integral(self):
        t = np.array([0.0, 1.0, 2.0])
        sensitivities = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        result = self._integral().postfix_sensitivities(
            "int y",
            np.ones(3),
            t,
            {"a": 1.0, "b": 2.0},
            ["a", "b"],
            sensitivities,
            boundary_sensitivities=np.array([[0.5, -0.5]]),
        )
        np.testing.assert_allclose(result, [[2.5, 3.5]])

    def test_boundary_sensitivities_of_the_wrong_size_raise(self):
        t = np.array([0.0, 1.0, 2.0])
        with pytest.raises(pybamm.ShapeError, match="boundary sensitivities"):
            self._integral().postfix_sensitivities(
                "int y",
                np.ones(3),
                t,
                {"a": 1.0, "b": 2.0},
                ["a", "b"],
                np.ones((3, 2)),
                boundary_sensitivities=np.zeros((1, 1)),
            )
