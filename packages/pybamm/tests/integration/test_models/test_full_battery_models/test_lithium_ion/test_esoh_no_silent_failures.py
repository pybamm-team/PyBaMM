#
# The electrode SOH solve must never report success with an answer that does not
# satisfy its own equations. This is the guard on the whole subsystem.
#
import numpy as np
import pytest

import pybamm

PARAMETER_SETS = [
    "Ai2020",
    "Chayambuka2022",
    "Chen2020",
    "Ecker2015",
    "Marquis2019",
    "Mohtat2020",
    "NCA_Kim2011",
    "OKane2022",
    "ORegan2022",
    "Prada2013",
    "Ramadass2004",
]
CAPACITY_STATES = [
    (1.0, 1.0),
    (0.8, 1.0),
    (1.0, 0.8),
    (1.2, 0.9),
    (0.9, 1.2),
    (0.7, 1.3),
    (1.3, 0.7),
]
INVENTORIES = 30
RESIDUAL_TOLERANCE = 1e-6


def _residuals(solver, solution, inputs):
    """The five equations the answer must satisfy, in volts and A.h."""
    parameter_values, param = solver.parameter_values, solver.param
    V_max = float(parameter_values.evaluate(param.ocp_soc_100))
    V_min = float(parameter_values.evaluate(param.ocp_soc_0))
    x_100, x_0 = float(solution["x_100"]), float(solution["x_0"])
    y_100, y_0 = float(solution["y_100"]), float(solution["y_0"])
    return {
        "V_max": float(solution["Up(y_100) - Un(x_100)"]) - V_max,
        "V_min": float(solution["Up(y_0) - Un(x_0)"]) - V_min,
        "Q_Li": x_100 * inputs["Q_n"] + y_100 * inputs["Q_p"] - inputs["Q_Li"],
        "Q_n": inputs["Q_n"] * (x_100 - x_0) - float(solution["Q"]),
        "Q_p": inputs["Q_p"] * (y_0 - y_100) - float(solution["Q"]),
    }


class TestElectrodeSOHNoSilentFailures:
    @pytest.mark.parametrize("parameter_set", PARAMETER_SETS)
    def test_a_returned_answer_always_satisfies_the_equations(self, parameter_set):
        parameter_values = pybamm.ParameterValues(parameter_set)
        solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values)
        param = solver.param
        Q_n = float(parameter_values.evaluate(param.n.Q_init))
        Q_p = float(parameter_values.evaluate(param.p.Q_init))
        x0_min, x100_max, y100_min, y0_max = solver.lims_ocp

        defects = []
        for scale_n, scale_p in CAPACITY_STATES:
            inputs = {"Q_n": Q_n * scale_n, "Q_p": Q_p * scale_p}
            low = inputs["Q_n"] * x0_min + inputs["Q_p"] * y100_min
            high = inputs["Q_n"] * x100_max + inputs["Q_p"] * y0_max
            for Q_Li in np.linspace(low, high, INVENTORIES + 2)[1:-1]:
                request = {**inputs, "Q_Li": float(Q_Li)}
                try:
                    solution = solver.solve(dict(request))
                except (pybamm.SolverError, ValueError):
                    continue  # refusing is always allowed; answering wrongly is not
                residuals = _residuals(solver, solution, request)
                worst = max(abs(value) for value in residuals.values())
                stoichiometries = [
                    float(solution[name]) for name in ("x_0", "x_100", "y_0", "y_100")
                ]
                if not all(-1e-9 <= v <= 1 + 1e-9 for v in stoichiometries):
                    defects.append((request["Q_Li"], "outside [0, 1]", stoichiometries))
                elif not np.isfinite(worst) or worst > RESIDUAL_TOLERANCE:
                    defects.append((request["Q_Li"], "residual", worst))

        assert not defects, (
            f"{len(defects)} of {len(CAPACITY_STATES) * INVENTORIES} answers for "
            f"{parameter_set} do not satisfy the equations: {defects[:3]}"
        )
