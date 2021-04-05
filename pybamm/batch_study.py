#
# BatchStudy class
#
import pybamm
from itertools import product


class BatchStudy:
    """
    A BatchStudy class for comparison of different PyBaMM simulations.

    Parameters
    ----------
    models : dict
        A dictionary of models to be simulated
    solvers : dict (optional)
        A dictionary of solvers to use to solve the model. Default is None
    experiments : dict (optional)
        A dictionary of experimental conditions under which to solve the model.
        Default is None
    repeats : int (optional)
        The number of times `solve` should be called. Default is 1
    permutations : bool (optional)
        If False runs first model with first solver, first experiment
        and second model with second solver, second experiment etc.
        If True runs a cartesian product of models, solvers and experiments.
        Default is False
    """

    def __init__(
        self, models, solvers=None, experiments=None, repeats=1, permutations=False
    ):
        self.models = models
        self.solvers = solvers
        self.experiments = experiments
        self.repeats = repeats
        self.permutations = permutations

        if not self.permutations:
            if self.solvers and (len(self.models) != len(self.solvers)):
                raise ValueError(
                    f"Either provide no solvers or an equal number of solvers as"
                    f"the models ({len(self.models)}) if permutations=False"
                )
            elif self.experiments and (len(self.models) != len(self.experiments)):
                raise ValueError(
                    f"Either provide no experiments or an equal number of experiments"
                    f"as the models ({len(self.models)}) if permutations=False"
                )

    def solve(self, t_eval=None):
        self.sims = []
        iter_func = product if self.permutations else zip

        # Instantiate values for solvers based on the value of self.permutations
        if self.solvers:
            solver_values = self.solvers.values()
        elif self.permutations:
            solver_values = [None]
        else:
            solver_values = [None] * len(self.models)

        # Instantiate values for experminents based on the value of self.permutations
        if self.experiments:
            experiment_values = self.experiments.values()
        elif self.permutations:
            experiment_values = [None]
        else:
            experiment_values = [None] * len(self.models)

        for model, solver, experiment in iter_func(
            self.models.values(), solver_values, experiment_values
        ):
            sim = pybamm.Simulation(model, solver=solver, experiment=experiment)
            # repeat to get average solve time and integration time
            solve_time = 0
            integration_time = 0
            for num in range(self.repeats):
                sol = sim.solve(t_eval)
                solve_time += sol.solve_time
                integration_time += sol.integration_time
            sim.solution.solve_time = solve_time / self.repeats
            sim.solution.integration_time = integration_time / self.repeats
            self.sims.append(sim)

    def plot(self, output_variables=None, **kwargs):
        self.quick_plot = pybamm.dynamic_plot(
            self.sims, output_variables=output_variables, **kwargs
        )
        return self.quick_plot
