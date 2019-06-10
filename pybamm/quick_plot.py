#
# Class for quick plotting of variables from models
#
import numpy as np
import pybamm
from collections import defaultdict


def ax_min(data):
    "Calculate appropriate minimum axis value for plotting"
    data_min = np.min(data)
    if data_min <= 0:
        return 1.1 * data_min
    else:
        return 0.9 * data_min


def ax_max(data):
    "Calculate appropriate maximum axis value for plotting"
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
    models: (iter of) :class:`pybamm.BaseModel`
        The model(s) to plot the outputs of.
    mesh: :class:`pybamm.Mesh`
        The mesh on which the model solved
    solutions: (iter of) :class:`pybamm.Solver`
        The numerical solution(s) for the model(s) which contained the solution to the
        model(s).
    output_variables : list of str, optional
        List of variables to plot
    labels : list of str, optional
        Labels for the different models. Defaults to model names
    """

    def __init__(self, models, mesh, solutions, output_variables=None, labels=None):
        # Pre-process models and solutions
        if isinstance(models, pybamm.BaseModel):
            models = [models]
        elif not isinstance(models, list):
            raise TypeError("'models' must be 'pybamm.BaseModel' or list")
        if isinstance(solutions, pybamm.Solution):
            solutions = [solutions]
        elif not isinstance(solutions, list):
            raise TypeError("'solutions' must be 'pybamm.Solution' or list")
        if len(models) == len(solutions):
            self.num_models = len(models)
        else:
            raise ValueError("must provide the same number of models and solutions")

        # Set labels
        self.labels = labels or [model.name for model in models]

        # Scales (default to 1 if information not in model)
        variables = models[0].variables
        self.x_scale = 1
        self.time_scale = 1
        if "x [m]" and "x" in variables:
            self.x_scale = (variables["x [m]"] / variables["x"]).evaluate()[-1]
        if "Time [m]" and "Time" in variables:
            self.time_scale = (variables["Time [h]"] / variables["Time"]).evaluate(t=1)

        # Time parameters
        self.ts = [solution.t for solution in solutions]
        self.min_t = np.min([t[0] for t in self.ts]) * self.time_scale
        self.max_t = np.max([t[-1] for t in self.ts]) * self.time_scale

        # Default output variables for lead-acid and lithium-ion
        if output_variables is None:
            if isinstance(models[0], pybamm.LithiumIonBaseModel):
                output_variables = [
                    "Negative particle surface concentration",
                    "Electrolyte concentration",
                    "Positive particle surface concentration",
                    "Current [A]",
                    "Negative electrode potential [V]",
                    "Electrolyte potential [V]",
                    "Positive electrode potential [V]",
                    "Terminal voltage [V]",
                ]
            elif isinstance(models[0], pybamm.LeadAcidBaseModel):
                output_variables = [
                    "Interfacial current density [A.m-2]",
                    "Electrolyte concentration [mol.m-3]",
                    "Current [A]",
                    "Porosity",
                    "Electrolyte potential [V]",
                    "Terminal voltage [V]",
                ]
            # else plot all variables in first model
            else:
                output_variables = models[0].variables

        self.set_output_variables(output_variables, solutions, models, mesh)
        self.reset_axis()

    def set_output_variables(self, output_variables, solutions, models, mesh):
        # Set up output variables
        self.variables = {}
        self.x_values = {}

        # Calculate subplot positions based on number of variables supplied
        self.subplot_positions = {}
        n = int(len(output_variables) // np.sqrt(len(output_variables)))
        m = np.ceil(len(output_variables) / n)

        # Process output variables into a form that can be plotted
        for k, variable_list in enumerate(output_variables):
            # Make sure we always have a list of lists of variables
            if isinstance(variable_list, str):
                variable_list = [variable_list]

            # Prepare list of variables
            key = tuple(variable_list)
            self.variables[key] = [None] * len(models)

            # process each variable in variable_list for each model
            for i, model in enumerate(models):
                # self.variables is a dictionary of lists of lists
                self.variables[key][i] = [
                    pybamm.ProcessedVariable(
                        model.variables[var], solutions[i].t, solutions[i].y, mesh
                    )
                    for var in variable_list
                ]

            # Make sure variables have the same dimensions and domain
            dim = self.variables[key][0][0].dimensions
            domain = self.variables[key][0][0].domain
            for variable in self.variables[key][0]:
                assert variable.domain == domain
                assert variable.dimensions == dim
            # Set the x-dimensions for
            if dim == 2:
                self.x_values[key] = mesh.combine_submeshes(*domain)[0].edges

            # Don't allow 3D variables
            elif dim == 3:
                raise NotImplementedError("cannot plot 3D variables")

            # Define subplot position
            self.subplot_positions[key] = (n, m, k + 1)

    def reset_axis(self):
        """
        Reset the axis limits to the default values.
        """
        self.axis = {}
        for key, variable_lists in self.variables.items():
            if variable_lists[0][0].dimensions == 1:
                y_min = np.min(
                    [
                        ax_min(var(self.ts[i]))
                        for i, variable_list in enumerate(variable_lists)
                        for var in variable_list
                    ]
                )
                y_max = np.max(
                    [
                        ax_max(var(self.ts[i]))
                        for i, variable_list in enumerate(variable_lists)
                        for var in variable_list
                    ]
                )
                if y_min == y_max:
                    y_min -= 1
                    y_max += 1
                self.axis[key] = [self.min_t, self.max_t, y_min, y_max]
            elif variable_lists[0][0].dimensions == 2:
                x = self.x_values[key]
                x_scaled = x * self.x_scale
                y_min = np.min(
                    [
                        ax_min(var(self.ts[i], x))
                        for i, variable_list in enumerate(variable_lists)
                        for var in variable_list
                    ]
                )
                y_max = np.max(
                    [
                        ax_max(var(self.ts[i], x))
                        for i, variable_list in enumerate(variable_lists)
                        for var in variable_list
                    ]
                )
                if y_min == y_max:
                    y_min -= 1
                    y_max += 1
                self.axis[key] = [x_scaled[0], x_scaled[-1], y_min, y_max]

    def plot(self, t):
        """Produces a quick plot with the internal states at time t.

        Parameters
        ----------
        t : float
            Dimensional time (in hours) at which to plot.
        """

        import matplotlib.pyplot as plt

        t /= self.time_scale
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        plt.tight_layout()
        plt.subplots_adjust(left=-0.1)
        self.plots = {}
        self.time_lines = {}

        for k, (key, variable_lists) in enumerate(self.variables.items()):
            plt.subplot(*self.subplot_positions[key])
            plt.ylabel(key, fontsize=14)
            plt.axis(self.axis[key])
            self.plots[key] = defaultdict(dict)
            # Set labels for the first subplot only (avoid repetition)
            if k == 0:
                labels = self.labels
            else:
                labels = [None] * len(self.labels)
            if variable_lists[0][0].dimensions == 2:
                # 2D plot: plot as a function of x at time t
                plt.xlabel("Position [m]", fontsize=14)
                x_value = self.x_values[key]
                for i, variable_list in enumerate(variable_lists):
                    for j, variable in enumerate(variable_list):
                        self.plots[key][i][j], = plt.plot(
                            x_value * self.x_scale,
                            variable(t, x_value),
                            lw=2,
                            label=labels[i],
                        )
            else:
                # 1D plot: plot as a function of time, indicating time t with a line
                plt.xlabel("Time [h]", fontsize=14)
                for i, variable_list in enumerate(variable_lists):
                    for j, variable in enumerate(variable_list):
                        full_t = self.ts[i]
                        self.plots[key][i][j], = plt.plot(
                            full_t * self.time_scale,
                            variable(full_t),
                            lw=2,
                            label=labels[i],
                        )
                        y_min, y_max = self.axis[key][2:]
                        self.time_lines[key], = plt.plot(
                            [t * self.time_scale, t * self.time_scale],
                            [y_min, y_max],
                            "k--",
                        )
        self.fig.legend(loc="lower right")

    def dynamic_plot(self, testing=False):
        """
        Generate a dynamic plot with a slider to control the time. We recommend using
        ipywidgets instead of this function if you are using jupyter notebooks
        """

        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        # create an initial plot at time 0
        self.plot(0)

        axcolor = "lightgoldenrodyellow"
        axfreq = plt.axes([0.315, 0.02, 0.37, 0.03], facecolor=axcolor)
        self.sfreq = Slider(axfreq, "Time", 0, self.max_t, valinit=0)
        self.sfreq.on_changed(self.update)

        plt.subplots_adjust(
            top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5
        )

        if not testing:  # pragma: no cover
            plt.show()

    def update(self, val):
        """
        Update the plot in self.plot() with values at new time
        """
        t = self.sfreq.val
        t_dimensionless = t / self.time_scale
        for key, plot in self.plots.items():
            if self.variables[key][0].dimensions == 2:
                x = self.x_values[key]
                for i, variable_lists in enumerate(self.variables[key]):
                    for j, variable in enumerate(variable_lists):
                        plot[i][j].set_ydata(variable(t_dimensionless, x))
            else:
                self.time_lines[key].set_xdata([t])

        self.fig.canvas.draw_idle()
