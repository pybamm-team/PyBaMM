#
# Class for quick plotting of variables from models
#
import numpy as np
import pybamm
import warnings
from collections import defaultdict


def ax_min(data):
    "Calculate appropriate minimum axis value for plotting"
    data_min = np.nanmin(data)
    if data_min <= 0:
        return 1.04 * data_min
    else:
        return 0.96 * data_min


def ax_max(data):
    "Calculate appropriate maximum axis value for plotting"
    data_max = np.nanmax(data)
    if data_max <= 0:
        return 0.96 * data_max
    else:
        return 1.04 * data_max


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


def get_spatial_scale(key, spatial_var_name, spatial_scales):
    "Return the appropriate spatial scale"
    if spatial_var_name == "r":
        if "negative" in key[0].lower():
            spatial_scale = spatial_scales["r_n"]
        elif "positive" in key[0].lower():
            spatial_scale = spatial_scales["r_p"]
    else:
        spatial_scale = spatial_scales[spatial_var_name]

    return spatial_scale


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
    meshes: (iter of) :class:`pybamm.Mesh`
        The mesh(es) on which the model(s) were solved.
    solutions: (iter of) :class:`pybamm.Solver`
        The numerical solution(s) for the model(s) which contained the solution to the
        model(s).
    output_variables : list of str, optional
        List of variables to plot
    labels : list of str, optional
        Labels for the different models. Defaults to model names
    colors : list of str, optional
        The colors to loop over when plotting. Defaults to
        ["r", "b", "k", "g", "m", "c"]
    linestyles : list of str, optional
        The linestyles to loop over when plotting. Defaults to ["-", ":", "--", "-."]
    """

    def __init__(
        self,
        solutions,
        output_variables=None,
        labels=None,
        colors=None,
        linestyles=None,
        figsize=None,
        time_format=None,
    ):
        if isinstance(solutions, pybamm.Solution):
            solutions = [solutions]
        elif not isinstance(solutions, list):
            raise TypeError("'solutions' must be 'pybamm.Solution' or list")

        models = [solution.model for solution in solutions]

        # Set labels
        if labels is None:
            self.labels = [model.name for model in models]
        else:
            if len(labels) != len(models):
                raise ValueError(
                    "labels '{}' have different length to models '{}'".format(
                        labels, [model.name for model in models]
                    )
                )
            self.labels = labels

        # Set colors, linestyles, figsize
        self.colors = colors or ["r", "b", "k", "g", "m", "c"]
        self.linestyles = linestyles or ["-", ":", "--", "-."]
        self.figsize = figsize or (15, 8)

        # Spatial scales (default to 1 if information not in model)
        variables = models[0].variables
        self.spatial_scales = {"x": 1, "y": 1, "z": 1, "r_n": 1, "r_p": 1}
        if "x [m]" and "x" in variables:
            self.spatial_scales["x"] = (variables["x [m]"] / variables["x"]).evaluate()[
                -1
            ]
        if "y [m]" and "y" in variables:
            self.spatial_scales["y"] = (variables["y [m]"] / variables["y"]).evaluate()[
                -1
            ]
        if "z [m]" and "z" in variables:
            self.spatial_scales["z"] = (variables["z [m]"] / variables["z"]).evaluate()[
                -1
            ]
        if "r_n [m]" and "r_n" in variables:
            self.spatial_scales["r_n"] = (
                variables["r_n [m]"] / variables["r_n"]
            ).evaluate()[-1]
        if "r_p [m]" and "r_p" in variables:
            self.spatial_scales["r_p"] = (
                variables["r_p [m]"] / variables["r_p"]
            ).evaluate()[-1]

        # Time parameters
        model_timescale_in_seconds = models[0].timescale_eval
        self.ts = [solution.t for solution in solutions]
        self.min_t = np.min([t[0] for t in self.ts]) * model_timescale_in_seconds
        self.max_t = np.max([t[-1] for t in self.ts]) * model_timescale_in_seconds

        # Set timescale
        if time_format is None:
            # defaults depend on how long the simulation is
            if self.max_t >= 3600:
                self.time_scale = model_timescale_in_seconds / 3600  # time in hours
            else:
                self.time_scale = model_timescale_in_seconds  # time in seconds
        elif time_format == "seconds":
            self.time_scale = model_timescale_in_seconds
        elif time_format == "minutes":
            self.time_scale = model_timescale_in_seconds / 60
        elif time_format == "hours":
            self.time_scale = model_timescale_in_seconds / 3600
        else:
            raise ValueError("time format '{}' not recognized".format(time_format))

        # Default output variables for lead-acid and lithium-ion
        if output_variables is None:
            if isinstance(models[0], pybamm.lithium_ion.BaseModel):
                output_variables = [
                    "Negative particle surface concentration [mol.m-3]",
                    "Electrolyte concentration [mol.m-3]",
                    "Positive particle surface concentration [mol.m-3]",
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

        self.set_output_variables(output_variables, solutions)
        self.reset_axis()

    def set_output_variables(self, output_variables, solutions):
        # Set up output variables
        self.variables = {}
        self.spatial_variable = {}

        # Calculate subplot positions based on number of variables supplied
        self.subplot_positions = {}
        self.n_rows = int(len(output_variables) // np.sqrt(len(output_variables)))
        self.n_cols = int(np.ceil(len(output_variables) / self.n_rows))

        # Prepare dictionary of variables
        # output_variables is a list of strings or lists, e.g.
        # ["var 1", ["variable 2", "var 3"]]
        for k, variable_list in enumerate(output_variables):
            # Make sure we always have a list of lists of variables, e.g.
            # [["var 1"], ["variable 2", "var 3"]]
            if isinstance(variable_list, str):
                variable_list = [variable_list]

            # Store the key as a tuple
            # key is the variable names, e.g. ("var 1",) or ("var 2", "var 3")
            key = tuple(variable_list)

            # Prepare list of variables
            variables = [None] * len(solutions)

            # process each variable in variable_list for each model
            for i, solution in enumerate(solutions):
                # variables lists of lists
                variables[i] = []
                # first index is the solution number
                # second index is the variable number
                for var in variable_list:
                    sol = solution[var]
                    # Check variable isn't all-nan
                    if np.all(np.isnan(sol.entries)):
                        raise ValueError("All-NaN variable '{}' provided".format(var))
                    # If ok, add to the list of solutions
                    else:
                        variables[i].append(sol)

            # Make sure variables have the same dimensions and domain
            # just use the first solution to check this
            first_solution = variables[0]
            first_variable = first_solution[0]
            domain = first_variable.domain
            # check all other variables against the first variable
            for idx, variable in enumerate(first_solution):
                if variable.domain != domain:
                    raise ValueError(
                        "Mismatching variable domains. "
                        "'{}' has domain '{}', but '{}' has domain '{}'".format(
                            key[0], domain, key[idx], variable.domain
                        )
                    )

            # Set the x variable (i.e. "x" or "r" for any one-dimensional variables)
            if first_variable.dimensions == 1:
                spatial_variable_key = first_variable.first_dimension
                spatial_variable_value = first_variable.first_dim_pts
                self.spatial_variable[key] = (
                    spatial_variable_key,
                    spatial_variable_value,
                )

            # Don't allow 2D variables if there are multiple solutions
            elif first_variable.dimensions == 2:
                if len(variables) > 1:
                    raise NotImplementedError(
                        "Cannot plot 3D variables when comparing multiple solutions, "
                        "but {} is 3D".format()
                    )
                else:
                    # Add both spatial variables to the keys
                    first_spatial_variable_key = first_variable.first_dimension
                    first_spatial_variable_value = first_variable.first_dim_pts
                    second_spatial_variable_key = first_variable.second_dimension
                    second_spatial_variable_value = first_variable.second_dim_pts
                    self.spatial_variable[key] = (
                        (first_spatial_variable_key, first_spatial_variable_value),
                        (second_spatial_variable_key, second_spatial_variable_value),
                    )

            # Store variables and subplot position
            self.variables[key] = variables
            self.subplot_positions[key] = (self.n_rows, self.n_cols, k + 1)

    def reset_axis(self):
        """
        Reset the axis limits to the default values.
        These are calculated to fit around the minimum and maximum values of all the
        variables in each subplot
        """
        self.axis = {}
        for key, variable_lists in self.variables.items():
            if variable_lists[0][0].dimensions == 0:
                spatial_var_name, spatial_var_value = "x", None
                x_min = self.min_t
                x_max = self.max_t
            elif variable_lists[0][0].dimensions == 1:
                spatial_var_name, spatial_var_value = self.spatial_variable[key]
                spatial_scale = get_spatial_scale(
                    key, spatial_var_name, self.spatial_scales
                )
                spatial_var_scaled = spatial_var_value * spatial_scale
                x_min = spatial_var_scaled[0]
                x_max = spatial_var_scaled[-1]
            elif variable_lists[0][0].dimensions == 2:
                spatial_vars = self.spatial_variable[key]
                # First spatial variable
                first_spatial_var_name, first_spatial_var_value = spatial_vars[0]
                first_spatial_scale = get_spatial_scale(
                    key, first_spatial_var_name, self.spatial_scales
                )
                first_spatial_var_scaled = first_spatial_var_value * first_spatial_scale
                x_min = first_spatial_var_scaled[0]
                x_max = first_spatial_var_scaled[-1]
                # Second spatial variable
                second_spatial_var_name, second_spatial_var_value = spatial_vars[1]
                second_spatial_scale = get_spatial_scale(
                    key, second_spatial_var_name, self.spatial_scales
                )
                second_spatial_var_scaled = (
                    second_spatial_var_value * second_spatial_scale
                )
                y_min = second_spatial_var_scaled[0]
                y_max = second_spatial_var_scaled[-1]
                self.axis[key] = [x_min, x_max, y_min, y_max]

            # Get min and max variable values
            if variable_lists[0][0].dimensions in [0, 1]:
                var_min = np.min(
                    [
                        ax_min(
                            var(
                                self.ts[i],
                                **{spatial_var_name: spatial_var_value},
                                warn=False
                            )
                        )
                        for i, variable_list in enumerate(variable_lists)
                        for var in variable_list
                    ]
                )
                var_max = np.max(
                    [
                        ax_max(
                            var(
                                self.ts[i],
                                **{spatial_var_name: spatial_var_value},
                                warn=False
                            )
                        )
                        for i, variable_list in enumerate(variable_lists)
                        for var in variable_list
                    ]
                )
                if var_min == var_max:
                    var_min -= 1
                    var_max += 1
                self.axis[key] = [x_min, x_max, var_min, var_max]

    def plot(self, t):
        """Produces a quick plot with the internal states at time t.

        Parameters
        ----------
        t : float
            Dimensional time (in hours) at which to plot.
        """

        import matplotlib.pyplot as plt

        t /= self.time_scale
        self.fig, self.ax = plt.subplots(self.n_rows, self.n_cols, figsize=self.figsize)
        plt.tight_layout()
        plt.subplots_adjust(left=-0.1)
        self.plots = {}
        self.time_lines = {}

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
            if variable_lists[0][0].dimensions == 0:
                # 0D plot: plot as a function of time, indicating time t with a line
                ax.set_xlabel("Time [h]", fontsize=fontsize)
                for i, variable_list in enumerate(variable_lists):
                    for j, variable in enumerate(variable_list):
                        full_t = self.ts[i]
                        (self.plots[key][i][j],) = ax.plot(
                            full_t * self.time_scale,
                            variable(full_t, warn=False),
                            lw=2,
                            color=self.colors[i],
                            linestyle=self.linestyles[j],
                        )
                y_min, y_max = self.axis[key][2:]
                (self.time_lines[key],) = ax.plot(
                    [t * self.time_scale, t * self.time_scale], [y_min, y_max], "k--"
                )
            elif variable_lists[0][0].dimensions == 1:
                # 1D plot: plot as a function of x at time t
                spatial_var_name, spatial_var_value = self.spatial_variable[key]
                ax.set_xlabel(spatial_var_name + " [m]", fontsize=fontsize)
                for i, variable_list in enumerate(variable_lists):
                    for j, variable in enumerate(variable_list):
                        spatial_scale = get_spatial_scale(
                            key, spatial_var_name, self.spatial_scales
                        )
                        (self.plots[key][i][j],) = ax.plot(
                            spatial_var_value * spatial_scale,
                            variable(
                                t, **{spatial_var_name: spatial_var_value}, warn=False
                            ),
                            lw=2,
                            color=self.colors[i],
                            linestyle=self.linestyles[j],
                        )
            elif variable_lists[0][0].dimensions == 2:
                # 2D plot: plot as a function of x and y at time t
                spatial_vars = self.spatial_variable[key]
                # first spatial variable
                first_spatial_var_name, first_spatial_var_value = spatial_vars[0]
                ax.set_xlabel(first_spatial_var_name + " [m]", fontsize=fontsize)
                first_spatial_scale = get_spatial_scale(
                    key, first_spatial_var_name, self.spatial_scales
                )
                # second spatial variable
                second_spatial_var_name, second_spatial_var_value = spatial_vars[1]
                ax.set_ylabel(second_spatial_var_name + " [m]", fontsize=fontsize)
                second_spatial_scale = get_spatial_scale(
                    key, second_spatial_var_name, self.spatial_scales
                )
                # there can only be one entry in the variable list
                variable = variable_lists[0][0]
                self.plots[key][0][0] = ax.contourf(
                    second_spatial_var_value * second_spatial_scale,
                    first_spatial_var_value * first_spatial_scale,
                    variable(
                        t,
                        **{
                            first_spatial_var_name: first_spatial_var_value,
                            second_spatial_var_name: second_spatial_var_value,
                        },
                        warn=False
                    ),
                )

            # Set either y label or legend entries
            if len(key) == 1:
                title = split_long_string(key[0])
                ax.set_title(title, fontsize=fontsize)
            else:
                ax.legend(
                    [split_long_string(s, 6) for s in key],
                    bbox_to_anchor=(0.5, 1),
                    fontsize=8,
                    loc="lower center",
                )

        # Set global legend
        # self.fig.legend(self.labels, loc="lower right")

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
        self.sfreq = Slider(axfreq, "Time [h]", 0, self.max_t, valinit=0)
        self.sfreq.on_changed(self.update)

        # ignore the warning about tight layout
        warnings.simplefilter("ignore")
        self.fig.tight_layout()
        warnings.simplefilter("always")

        if not testing:  # pragma: no cover
            plt.show()

    def update(self, val):
        """
        Update the plot in self.plot() with values at new time
        """
        t = self.sfreq.val
        t_dimensionless = t / self.time_scale
        for key, plot in self.plots.items():
            if self.variables[key][0][0].dimensions == 0:
                self.time_lines[key].set_xdata([t])
            if self.variables[key][0][0].dimensions == 1:
                spatial_var_name, spatial_var_value = self.spatial_variable[key]
                for i, variable_lists in enumerate(self.variables[key]):
                    for j, variable in enumerate(variable_lists):
                        plot[i][j].set_ydata(
                            variable(
                                t_dimensionless,
                                **{spatial_var_name: spatial_var_value},
                                warn=False
                            )
                        )

        self.fig.canvas.draw_idle()
