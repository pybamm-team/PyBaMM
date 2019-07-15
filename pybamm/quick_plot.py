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


def split_long_string(title, max_words=4):
    "Get title in a nice format"
    words = title.split()
    # Don't split if fits on one line, don't split just for units
    if len(words) <= max_words or words[max_words].startswith("["):
        return title
    else:
        first_line = (" ").join(words[:max_words])
        second_line = (" ").join(words[max_words:])
        return first_line + "\n" + second_line


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

    def __init__(self, models, mesh, solutions, output_variables=None):
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
        self.labels = [model.name for model in models]

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
            if isinstance(models[0], pybamm.lithium_ion.BaseModel):
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
            elif isinstance(models[0], pybamm.lead_acid.BaseModel):
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
        self.n_rows = int(len(output_variables) // np.sqrt(len(output_variables)))
        self.n_cols = int(np.ceil(len(output_variables) / self.n_rows))

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
            domain = self.variables[key][0][0].domain
            for variable in self.variables[key][0]:
                if variable.domain != domain:
                    raise ValueError("mismatching variable domains")

            # Set the x variable for any two-dimensional variables
            if self.variables[key][0][0].dimensions == 2:
                self.x_values[key] = mesh.combine_submeshes(*domain)[0].edges

            # Don't allow 3D variables
            elif any(var.dimensions == 3 for var in self.variables[key][0]):
                raise NotImplementedError("cannot plot 3D variables")

            # Define subplot position
            self.subplot_positions[key] = (self.n_rows, self.n_cols, k + 1)

    def reset_axis(self):
        """
        Reset the axis limits to the default values.
        These are calculated to fit around the minimum and maximum values of all the
        variables in each subplot
        """
        self.axis = {}
        for key, variable_lists in self.variables.items():
            if variable_lists[0][0].dimensions == 1:
                x = None
                x_min = self.min_t
                x_max = self.max_t
            elif variable_lists[0][0].dimensions == 2:
                x = self.x_values[key]
                x_scaled = x * self.x_scale
                x_min = x_scaled[0]
                x_max = x_scaled[-1]

            # Get min and max y values
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
            self.axis[key] = [x_min, x_max, y_min, y_max]

    def plot(self, t, dynamic=True, figsize=(15, 8)):
        """Produces a quick plot with the internal states at time t.

        Parameters
        ----------
        t : float
            Dimensional time (in hours) at which to plot.
        """

        import matplotlib.pyplot as plt

        t /= self.time_scale
        self.fig, self.ax = plt.subplots(self.n_rows, self.n_cols, figsize=figsize)
        plt.tight_layout()
        plt.subplots_adjust(left=-0.1)
        self.plots = {}
        self.time_lines = {}

        colors = ["k-", "g--", "r:", "b-."]  # ["k", "g", "r", "b"]
        linestyles = ["-", "--", ":", "-."]
        fontsize = 42 // self.n_cols

        for k, (key, variable_lists) in enumerate(self.variables.items()):
            if len(self.variables) == 1:
                ax = self.ax
            else:
                ax = self.ax.flat[k]
            ax.set_xlim(self.axis[key][:2])
            ax.set_ylim(self.axis[key][2:])
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            self.plots[key] = defaultdict(dict)
            # Set labels for the first subplot only (avoid repetition)
            if k == 0:
                labels = self.labels
            else:
                labels = [None] * len(self.labels)
            if variable_lists[0][0].dimensions == 2:
                # 2D plot: plot as a function of x at time t
                ax.set_xlabel("Position [m]", fontsize=fontsize)
                x_value = self.x_values[key]
                for i, variable_list in enumerate(variable_lists):
                    for j, variable in enumerate(variable_list):
                        if j == 0:
                            label = labels[i]
                        else:
                            label = None
                        self.plots[key][i][j], = ax.plot(
                            x_value * self.x_scale,
                            variable(t, x_value),
                            colors[i],
                            lw=2,
                            # linestyle=linestyles[j],
                            label=label,
                        )
            else:
                # 1D plot: plot as a function of time, indicating time t with a line
                ax.set_xlabel("Time [h]", fontsize=fontsize)
                for i, variable_list in enumerate(variable_lists):
                    for j, variable in enumerate(variable_list):
                        full_t = self.ts[i]
                        if j == 0:
                            label = labels[i]
                        else:
                            label = None
                        self.plots[key][i][j], = ax.plot(
                            full_t * self.time_scale,
                            variable(full_t),
                            colors[i],
                            lw=2,
                            # linestyle=linestyles[j],
                            label=label,
                        )
                y_min, y_max = self.axis[key][2:]
                self.time_lines[key], = ax.plot(
                    [t * self.time_scale, t * self.time_scale], [y_min, y_max], "k--"
                )
            # Set either y label or legend entries
            if len(key) == 1:
                title = split_long_string(key[0])
                ax.set_title(title, fontsize=fontsize)
            else:
                ax.legend(
                    [split_long_string(s, 6) for s in key],
                    bbox_to_anchor=(0.5, 1.2),
                    fontsize=8,
                    loc="upper center",
                )
        if dynamic is True:
            self.fig.legend(loc="lower right", fontsize=21)
        else:
            ax.legend(self.labels, bbox_to_anchor=(1.05, 1.5), loc=2, fontsize=21)

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
            if self.variables[key][0][0].dimensions == 2:
                x = self.x_values[key]
                for i, variable_lists in enumerate(self.variables[key]):
                    for j, variable in enumerate(variable_lists):
                        plot[i][j].set_ydata(variable(t_dimensionless, x))
            else:
                self.time_lines[key].set_xdata([t])

        self.fig.canvas.draw_idle()
