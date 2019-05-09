import numpy as np
import pybamm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def ax_min(data):
    data_min = np.min(data)
    if data_min <= 0:
        return 1.1 * data_min
    else:
        return 0.9 * data_min


def ax_max(data):
    data_max = np.max(data)
    if data_max <= 0:
        return 0.9 * data_max
    else:
        return 1.1 * data_max


class QuickPlot(object):
    """
    Generates a quick plot of a subset of key outputs of the model so that the model
    outputs can be easily assessed. The axis limits can be set using:
        self.axis["Variable name"] = [x_min, x_max, y_min, y_max]
    They can be reset to the default values by using self.reset_axis.

    Parameters
    ----------
    model: :class: pybamm.BaseModel
        The model to plot the outputs of.
    mesh: :class: pybamm.Mesh
        The mesh on which the model solved
    solver: :class: pybamm.Solver
        The numerical solver for the model which contained the solution to the model.
    """

    def __init__(self, models, param, mesh, solvers, output_variables=None):
        if isinstance(models, pybamm.BaseModel):
            models = [models]
        elif not isinstance(models, list):
            raise TypeError("'models' must be 'pybamm.BaseModel' or list")
        if isinstance(solvers, pybamm.BaseSolver):
            solvers = [solvers]
        elif not isinstance(solvers, list):
            raise TypeError("'solvers' must be 'pybamm.BaseSolver' or list")
        if len(models) == len(solvers):
            self.num_models = len(models)
        else:
            raise ValueError("must provide the same number of models and solutions")

        # Time parameters
        self.ts = [solver.t for solver in solvers]
        self.min_t = np.min([t[0] for t in self.ts])
        self.max_t = np.max([t[-1] for t in self.ts])

        # Default output variables for lead-acid and lithium-ion
        if output_variables is None:
            if isinstance(models[0], pybamm.LithiumIonBaseModel):
                output_variables = {
                    "Negative particle surface concentration": ((2, 4, 1), "x"),
                    "Electrolyte concentration": ((2, 4, 2), "x"),
                    "Positive particle surface concentration": ((2, 4, 3), "x"),
                    "Total current density": ((2, 4, 4), "t"),
                    "Negative electrode potential [V]": ((2, 4, 5), "x"),
                    "Electrolyte potential [V]": ((2, 4, 6), "x"),
                    "Positive electrode potential [V]": ((2, 4, 7), "x"),
                    "Terminal voltage [V]": ((2, 4, 8), "t"),
                }
            elif isinstance(models[0], pybamm.LeadAcidBaseModel):
                output_variables = {
                    "Interfacial current density": ((2, 3, 1), "x"),
                    "Electrolyte concentration": ((2, 3, 2), "x"),
                    "Total current density": ((2, 3, 3), "t"),
                    "Porosity": ((2, 3, 4), "x"),
                    "Electrolyte potential [V]": ((2, 3, 5), "x"),
                    "Terminal voltage [V]": ((2, 3, 6), "t"),
                }
        self.set_output_variables(output_variables, solvers, models, mesh)
        self.reset_axis()

    def set_output_variables(self, output_variables, solvers, models, mesh):
        # Set up output variables
        self.variables = {}
        self.x_values = {}
        self.subplot_positions = {}
        self.independent_variables = {}
        for var, (subplot_position, indep_var) in output_variables.items():
            self.variables[var] = [
                pybamm.ProcessedVariable(
                    models[i].variables[var], solvers[i].t, solvers[i].y, mesh
                )
                for i in range(len(models))
            ]
            domain = models[0].variables[var].domain
            self.subplot_positions[var] = subplot_position
            self.independent_variables[var] = indep_var
            if indep_var == "x":
                self.x_values[var] = mesh.combine_submeshes(*domain)[0].edges

    def reset_axis(self):
        """
        Reset the axis limits to the default values.
        """
        self.axis = {}
        for name, variable in self.variables.items():
            if self.independent_variables[name] == "x":
                x = self.x_values[name]
                y_min = np.min(
                    [ax_min(v(self.ts[i], x)) for i, v in enumerate(variable)]
                )
                y_max = np.max(
                    [ax_max(v(self.ts[i], x)) for i, v in enumerate(variable)]
                )
                self.axis[name] = [x[0], x[-1], y_min, y_max]
            elif self.independent_variables[name] == "t":
                y_min = np.min([ax_min(v(self.ts[i])) for i, v in enumerate(variable)])
                y_max = np.max([ax_max(v(self.ts[i])) for i, v in enumerate(variable)])
                self.axis[name] = [self.min_t, self.max_t, y_min, y_max]

    def plot(self, t):
        """Produces a quick plot with the internal states at time t.

        Parameters
        ----------
        t : float
            Time at which to plot.
        """
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        plt.tight_layout()
        plt.subplots_adjust(left=-0.1)
        self.plots = {}
        self.time_lines = {}

        for name, variable in self.variables.items():
            plt.subplot(*self.subplot_positions[name])
            plt.ylabel(name)
            plt.axis(self.axis[name])
            self.plots[name] = [None] * self.num_models
            if self.independent_variables[name] == "x":
                plt.xlabel("Position")
                x_value = self.x_values[name]
                for i in range(self.num_models):
                    self.plots[name][i], = plt.plot(
                        x_value, variable[i](t, x_value), lw=2
                    )
            else:
                plt.xlabel("Time")
                for i in range(self.num_models):
                    full_t = self.ts[i]
                    self.plots[name][i], = plt.plot(full_t, variable[i](full_t), lw=2)
                    y_min, y_max = self.axis[name][2:]
                    self.time_lines[name], = plt.plot([t, t], [y_min, y_max], "k--")

    def dynamic_plot(self, testing=False):
        """
        Generate a dynamic plot with a slider to control the time. We recommend using
        ipywidgets instead of this function if you are using jupyter notebooks
        """

        # create an initial plot at time 0
        self.plot(0)

        axcolor = "lightgoldenrodyellow"
        axfreq = plt.axes([0.315, 0.05, 0.37, 0.03], facecolor=axcolor)
        self.sfreq = Slider(axfreq, "Time", 0, self.max_t, valinit=0)
        self.sfreq.on_changed(self.update)

        plt.subplots_adjust(
            top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5
        )

        if not testing:
            plt.show()

    def update(self, val):
        """
        Update the plot in self.plot() with values at new time
        """
        t = self.sfreq.val
        for var, plot in self.plots.items():
            if self.independent_variables[var] == "x":
                x = self.x_values[var]
                for i in range(self.num_models):
                    plot[i].set_ydata(self.variables[var][i](t, x))
            else:
                self.time_lines[var].set_xdata([t])

        self.fig.canvas.draw_idle()
