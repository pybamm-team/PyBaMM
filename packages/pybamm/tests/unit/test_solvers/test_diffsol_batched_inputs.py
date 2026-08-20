"""Diffsol with several input sets: one solution per set.

``BaseSolver`` fans multiple input sets out across a process pool, which diffsol
cannot use — ``PreparedSolver`` is a PyO3 object and will not pickle. The solver
runs its own fan-out instead (see ``TestDiffsolNumThreads`` in
``test_diffsol_solver.py``), defaulting to a serial loop, and either way must
honour the contract of returning one result per input set.
"""

import numpy as np

import pybamm

_SOLVER_TOL = 1e-8
_CURRENTS = [0.5, 1.5]


class TestDiffsolBatchedInputs:
    def _build(self):
        model = pybamm.lithium_ion.SPM()
        params = model.default_parameter_values
        params["Current function [A]"] = "[input]"
        sim = pybamm.Simulation(
            model,
            parameter_values=params,
            solver=pybamm.DiffsolSolver(rtol=_SOLVER_TOL, atol=_SOLVER_TOL),
        )
        sim.build()
        return sim

    def test_returns_one_solution_per_input_set(self):
        sim = self._build()
        t_eval = np.linspace(0, 300, 11)
        inputs = [{"Current function [A]": current} for current in _CURRENTS]

        solutions = sim.solver.solve(sim.built_model, t_eval, inputs=inputs)

        assert isinstance(solutions, list)
        assert len(solutions) == len(inputs)

    def test_each_solution_carries_its_own_input_set(self):
        sim = self._build()
        t_eval = np.linspace(0, 300, 11)
        inputs = [{"Current function [A]": current} for current in _CURRENTS]

        solutions = sim.solver.solve(sim.built_model, t_eval, inputs=inputs)

        for solution, current in zip(solutions, _CURRENTS, strict=True):
            assert solution.all_inputs[0]["Current function [A]"] == np.array([current])

    def test_distinct_input_sets_give_distinct_trajectories(self):
        sim = self._build()
        t_eval = np.linspace(0, 300, 11)
        inputs = [{"Current function [A]": current} for current in _CURRENTS]

        solutions = sim.solver.solve(sim.built_model, t_eval, inputs=inputs)

        first = solutions[0]["Voltage [V]"](t_eval)
        second = solutions[1]["Voltage [V]"](t_eval)
        # A 3x larger current must discharge measurably faster.
        assert np.all(second < first)
