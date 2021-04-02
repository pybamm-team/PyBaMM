#
# BatchStudy class
#
import pickle
import pybamm
from itertools import product


class BatchStudy:
    """
    comparison class
    """

    def __init__(
        self, models, solvers=None, experiments=None, repeats=1, permutations=False
    ):

        self.models = models
        self.solvers = solvers
        self.experiments = experiments
        self.repeats = repeats
        self.permutations = permutations

    def solve(self):
        self.sims = []
        for num in range(self.repeats):
            if self.permutations is False:
                for model, solver, experiment in zip(
                    self.models.values(),
                    self.solvers.values(),
                    self.experiments.values(),
                ):
                    sim = pybamm.Simulation(model, solver=solver, experiment=experiment)
                    sim.solve([0, 3600])
                    self.sims.append(sim)
            else:
                for values in product(
                    self.models.values(),
                    self.solvers.values(),
                    self.experiments.values(),
                ):
                    model, solver, experiment = values
                    sim = pybamm.Simulation(model, solver=solver, experiment=experiment)
                    sim.solve([0, 3600])
                    self.sims.append(sim)

    def plot(self, output_variables=None, quick_plot_vars=None, **kwargs):
        pybamm.dynamic_plot(self.sims)

    def save(self, filename):  #TODO
        return pybamm.Simulation().save(filename)
