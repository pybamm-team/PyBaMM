#
# Convergence of the electrode SOH solve across the whole usable range
#
import numpy as np
import pytest

import pybamm

# Every shipped full-cell parameter set with plain (non-MSMR) open-circuit potentials.
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
    "Xu2019",
]

# 12 sets x 5 capacity states x 20 lithium inventories = 1200 solves
CAPACITY_STATES = [(1.0, 1.0), (0.8, 1.0), (1.0, 0.8), (1.2, 0.9), (0.9, 1.2)]
INVENTORIES = 20

# The answer must satisfy the model's own equations, not merely fail to raise.
RESIDUAL_TOLERANCE = 1e-10


def _residuals(solver, solution, inputs):
    """The three equations the electrode SOH solution must satisfy, in volts and A.h."""
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


def _sweep(parameter_set):
    """Every (capacity state, lithium inventory) point for one parameter set."""
    parameter_values = pybamm.ParameterValues(parameter_set)
    solver = pybamm.lithium_ion.ElectrodeSOHSolver(parameter_values)
    param = solver.param
    Q_n = float(parameter_values.evaluate(param.n.Q_init))
    Q_p = float(parameter_values.evaluate(param.p.Q_init))

    for scale_n, scale_p in CAPACITY_STATES:
        inputs = {"Q_n": Q_n * scale_n, "Q_p": Q_p * scale_p}
        x0_min, x100_max, y100_min, y0_max = solver.lims_ocp
        low = inputs["Q_n"] * x0_min + inputs["Q_p"] * y100_min
        high = inputs["Q_n"] * x100_max + inputs["Q_p"] * y0_max
        # strictly inside, so _get_lims accepts every point as feasible
        for Q_Li in np.linspace(low, high, INVENTORIES + 2)[1:-1]:
            yield solver, {**inputs, "Q_Li": float(Q_Li)}


class TestElectrodeSOHConvergence:
    """
    Every feasible electrode SOH request must either solve correctly or raise. A
    returned answer that does not satisfy the equations is the failure this guards.
    """

    @pytest.mark.parametrize("parameter_set", PARAMETER_SETS)
    def test_every_feasible_state_converges(self, parameter_set):
        failures, worst = [], 0.0
        for solver, inputs in _sweep(parameter_set):
            try:
                solution = solver.solve(dict(inputs))
            except (pybamm.SolverError, ValueError):
                continue  # an infeasible request is allowed to raise
            residuals = _residuals(solver, solution, inputs)
            stoichiometries = [
                float(solution[name]) for name in ("x_0", "x_100", "y_0", "y_100")
            ]
            worst_residual = max(abs(value) for value in residuals.values())
            if not all(-1e-9 <= value <= 1 + 1e-9 for value in stoichiometries):
                failures.append((inputs["Q_Li"], f"outside [0, 1]: {stoichiometries}"))
            elif not np.isfinite(worst_residual) or worst_residual > RESIDUAL_TOLERANCE:
                failures.append((inputs["Q_Li"], f"residuals {residuals}"))
            else:
                worst = max(worst, worst_residual)

        assert not failures, (
            f"{len(failures)} of {len(CAPACITY_STATES) * INVENTORIES} points failed "
            f"for {parameter_set}: {failures[:3]}"
        )
        assert worst < RESIDUAL_TOLERANCE, worst

    @pytest.mark.parametrize(
        # points where the solve returns "success" with a large residual on main
        ("parameter_set", "inputs"),
        [
            ("Ai2020", {"Q_n": 3.8030, "Q_p": 3.2194, "Q_Li": 3.7669}),
            ("Ramadass2004", {"Q_n": 2.1349, "Q_p": 3.9431, "Q_Li": 2.4312}),
        ],
    )
    def test_known_bad_points(self, parameter_set, inputs):
        solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            pybamm.ParameterValues(parameter_set)
        )
        solution = solver.solve(dict(inputs))
        residuals = _residuals(solver, solution, inputs)
        assert max(abs(value) for value in residuals.values()) < RESIDUAL_TOLERANCE

    def test_an_infeasible_request_raises(self):
        solver = pybamm.lithium_ion.ElectrodeSOHSolver(
            pybamm.ParameterValues("Chen2020")
        )
        param = solver.param
        Q_n = float(solver.parameter_values.evaluate(param.n.Q_init))
        Q_p = float(solver.parameter_values.evaluate(param.p.Q_init))
        with pytest.raises(ValueError, match="outside the range of possible values"):
            solver.solve({"Q_n": Q_n, "Q_p": Q_p, "Q_Li": 100 * (Q_n + Q_p)})
