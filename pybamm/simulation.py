#
# Simulation class for a battery model
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

import argparse


class Simulation(object):
    """
    The simulation class for a battery model.

    Parameters
    ---------
    model : pybamm.models.(modelname).(ModelName)() instance
       The model to be used for the simulation. (modelname) and (ModelName)
       refer to a module and class to be chosen.
    parameter_values : :class:`pybamm.ParameterValues.Parameters` instance
       The parameters to be used for the simulation.
    discretisation : :class:`pybamm.discretisation.Mesh` instance
       The discretisation to be used for the simulation.
    solver : :class:`pybamm.solver.Solver` instance
       The algorithm for solving the model.
    name : string, optional
       The simulation name.

    """

    def __init__(
        self,
        model,
        parameter_values=None,
        discretisation=None,
        solver=None,
        name="unnamed",
    ):
        # Read defaults from model
        if parameter_values is None:
            parameter_values = model.default_parameter_values
        if discretisation is None:
            discretisation = model.default_discretisation
        if solver is None:
            solver = model.default_solver

        # Assign attributes
        self.model = model
        self.parameter_values = parameter_values
        self.discretisation = discretisation
        self.solver = solver
        self.name = name

    def __str__(self):
        return self.name

    def set_parameters(self):
        self.parameter_values.process_model(self.model)

    def discretise(self):
        self.discretisation.process_model(self.model)

    def solve(self):
        self.solver.solve(self.model, self.discretisation.mesh["time"])

    def run(self):
        self.set_parameters()
        self.discretise()
        self.solve()

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError


if __name__ == "__main_":
    # Read inputs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", nargs="?", default="DFN", help="the model to be run"
    )
    parser.add_argument(
        "--current", type=float, nargs=1, help="the charge/discharge current"
    )
    parser.add_argument(
        "--Crate", type=float, nargs=1, help="the charge/discharge C-rate"
    )
    # parser.add_argument("-s", "--save", action="store_true", help="save the output")
    # parser.add_argument(
    #     "-f",
    #     "--force",
    #     action="store_true",
    #     help="overwrite saved output even if it is available",
    # )
    args = parser.parse_args()

    model = getattr(pybamm, args.model_name)
    sim = Simulation(model)
    sim.run()
