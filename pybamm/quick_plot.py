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
    figsize : tuple of floats
        The size of the figure to make
    time_format : str
        Format for the time output ("hours", "minutes" or "seconds")
    spatial_format : str
        Format for the spatial axes ("m", "mm" or "um")

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
        spatial_format="m",
    ):
        if isinstance(solutions, (pybamm.Solution, pybamm.Simulation)):
            solutions = [solutions]
        elif not isinstance(solutions, list):
            raise TypeError(
                "solutions must be 'pybamm.Solution' or 'pybamm.Simulation' or list"
            )

        # Extract solution from any simulations
        for idx, sol in enumerate(solutions):
            if isinstance(sol, pybamm.Simulation):
                # 'sol' is actually a 'Simulation' object here so it has a 'Solution'
                # attribute
                solutions[idx] = sol.solution

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
        if spatial_format == "m":
            spatial_factor = 1
        elif spatial_format == "mm":
            spatial_factor = 1e3
        elif spatial_format == "um":  # micrometers
            spatial_factor = 1e6
        else:
            raise ValueError(
                "spatial format '{}' not recognized".format(spatial_format)
            )

        self.spatial_format = spatial_format

        variables = models[0].variables
        self.spatial_scales = {"x": 1, "y": 1, "z": 1, "r_n": 1, "r_p": 1}
        if "x [m]" and "x" in variables:
            self.spatial_scales["x"] = (variables["x [m]"] / variables["x"]).evaluate()[
                -1
            ] * spatial_factor
        if "y [m]" and "y" in variables:
            self.spatial_scales["y"] = (variables["y [m]"] / variables["y"]).evaluate()[
                -1
            ] * spatial_factor
        if "z [m]" and "z" in variables:
            self.spatial_scales["z"] = (variables["z [m]"] / variables["z"]).evaluate()[
                -1
            ] * spatial_factor
        if "r_n [m]" and "r_n" in variables:
            self.spatial_scales["r_n"] = (
                variables["r_n [m]"] / variables["r_n"]
            ).evaluate()[-1] * spatial_factor
        if "r_p [m]" and "r_p" in variables:
            self.spatial_scales["r_p"] = (
                variables["r_p [m]"] / variables["r_p"]
            ).evaluate()[-1] * spatial_factor

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
        self.spatial_variable_dict = {}
        self.first_dimensional_spatial_variable = {}
        self.second_dimensional_spatial_variable = {}

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
                # variables lists of lists, so variables[i] is a list
                variables[i] = []
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
            # check all other solutions against the first solution
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
                (
                    spatial_var_name,
                    spatial_var_value,
                    spatial_var_value_dimensional,
                ) = self.get_spatial_var(key, first_variable, "first")
                self.spatial_variable_dict[key] = {spatial_var_name: spatial_var_value}
                self.first_dimensional_spatial_variable[
                    key
                ] = spatial_var_value_dimensional

            elif first_variable.dimensions == 2:
                # Don't allow 2D variables if there are multiple solutions
                if len(variables) > 1:
                    raise NotImplementedError(
                        "Cannot plot 2D variables when comparing multiple solutions, "
                        "but {} is 2D".format(key[0])
                    )
                # But do allow if just a single solution
                else:
                    # Add both spatial variables to the keys
                    (
                        first_spatial_var_name,
                        first_spatial_var_value,
                        first_spatial_var_value_dimensional,
                    ) = self.get_spatial_var(key, first_variable, "first")
                    (
                        second_spatial_var_name,
                        second_spatial_var_value,
                        second_spatial_var_value_dimensional,
                    ) = self.get_spatial_var(key, first_variable, "second")
                    self.spatial_variable_dict[key] = {
                        first_spatial_var_name: first_spatial_var_value,
                        second_spatial_var_name: second_spatial_var_value,
                    }
                    self.first_dimensional_spatial_variable[
                        key
                    ] = first_spatial_var_value_dimensional
                    self.second_dimensional_spatial_variable[
                        key
                    ] = second_spatial_var_value_dimensional

            # Store variables and subplot position
            self.variables[key] = variables
            self.subplot_positions[key] = (self.n_rows, self.n_cols, k + 1)

    def get_spatial_var(self, key, variable, dimension):
        "Return the appropriate spatial variable(s)"

        # Extract name and dimensionless value
        if dimension == "first":
            spatial_var_name = variable.first_dimension
            spatial_var_value = variable.first_dim_pts
        elif dimension == "second":
            spatial_var_name = variable.second_dimension
            spatial_var_value = variable.second_dim_pts

        # Get scale
        if spatial_var_name == "r":
            if "negative" in key[0].lower():
                spatial_scale = self.spatial_scales["r_n"]
            elif "positive" in key[0].lower():
                spatial_scale = self.spatial_scales["r_p"]
            else:
                raise NotImplementedError(
                    "Cannot determine the spatial scale for '{}'".format(key[0])
                )
        else:
            spatial_scale = self.spatial_scales[spatial_var_name]

        # Get dimensional variable
        spatial_var_value_dim = spatial_var_value * spatial_scale

        return spatial_var_name, spatial_var_value, spatial_var_value_dim

    def reset_axis(self):
        """
        Reset the axis limits to the default values.
        These are calculated to fit around the minimum and maximum values of all the
        variables in each subplot
        """
        self.axis = {}
        for key, variable_lists in self.variables.items():
            if variable_lists[0][0].dimensions == 0:
                x_min = self.min_t
                x_max = self.max_t
                spatial_vars = {}
            elif variable_lists[0][0].dimensions == 1:
                x_min = self.first_dimensional_spatial_variable[key][0]
                x_max = self.first_dimensional_spatial_variable[key][-1]
                # Read dictionary of spatial variables
                spatial_vars = self.spatial_variable_dict[key]
            elif variable_lists[0][0].dimensions == 2:
                # First spatial variable
                x_min = self.first_dimensional_spatial_variable[key][0]
                x_max = self.first_dimensional_spatial_variable[key][-1]
                y_min = self.second_dimensional_spatial_variable[key][0]
                y_max = self.second_dimensional_spatial_variable[key][-1]

                # Read dictionary of spatial variables
                spatial_vars = self.spatial_variable_dict[key]

                # Create axis for contour plot
                self.axis[key] = [x_min, x_max, y_min, y_max]

            # Get min and max variable values
            var_min = np.min(
                [
                    ax_min(var(self.ts[i], **spatial_vars, warn=False))
                    for i, variable_list in enumerate(variable_lists)
                    for var in variable_list
                ]
            )
            var_max = np.max(
                [
                    ax_max(var(self.ts[i], **spatial_vars, warn=False))
                    for i, variable_list in enumerate(variable_lists)
                    for var in variable_list
                ]
            )
            if var_min == var_max:
                var_min -= 1
                var_max += 1
            if variable_lists[0][0].dimensions in [0, 1]:
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

        if self.n_cols == 1:
            fontsize = 30
        else:
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
                # Read dictionary of spatial variables
                spatial_vars = self.spatial_variable_dict[key]
                spatial_var_name = list(spatial_vars.keys())[0]
                ax.set_xlabel(
                    "{} [{}]".format(spatial_var_name, self.spatial_format),
                    fontsize=fontsize,
                )
                for i, variable_list in enumerate(variable_lists):
                    for j, variable in enumerate(variable_list):
                        (self.plots[key][i][j],) = ax.plot(
                            self.first_dimensional_spatial_variable[key],
                            variable(t, **spatial_vars, warn=False),
                            lw=2,
                            color=self.colors[i],
                            linestyle=self.linestyles[j],
                        )
            elif variable_lists[0][0].dimensions == 2:
                # 2D plot: plot as a function of x and y at time t
                # Read dictionary of spatial variables
                spatial_vars = self.spatial_variable_dict[key]
                x_name = list(spatial_vars.keys())[0][0]
                y_name = list(spatial_vars.keys())[1][0]
                ax.set_xlabel(
                    "{} [{}]".format(x_name, self.spatial_format), fontsize=fontsize
                )
                ax.set_ylabel(
                    "{} [{}]".format(y_name, self.spatial_format), fontsize=fontsize
                )
                # there can only be one entry in the variable list
                variable = variable_lists[0][0]
                self.plots[key][0][0] = ax.contourf(
                    self.second_dimensional_spatial_variable[key],
                    self.first_dimensional_spatial_variable[key],
                    variable(t, **spatial_vars, warn=False),
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

    def dynamic_plot(self, testing=False, step=None):
        """
        Generate a dynamic plot with a slider to control the time.

        Parameters
        ----------
        step : float
            For notebook mode, size of steps to allow in the slider. Defaults to 1/100th
            of the total time.
        testing : bool
            Whether to actually make the plot (turned off for unit tests)

        """
        if pybamm.is_notebook():  # pragma: no cover
            import ipywidgets as widgets

            step = step or self.max_t / 100
            widgets.interact(
                self.plot,
                t=widgets.FloatSlider(min=0, max=self.max_t, step=step, value=0),
                continuous_update=False,
            )
        else:
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
                for i, variable_lists in enumerate(self.variables[key]):
                    for j, variable in enumerate(variable_lists):
                        plot[i][j].set_ydata(
                            variable(
                                t_dimensionless,
                                **self.spatial_variable_dict[key],
                                warn=False
                            )
                        )

        self.fig.canvas.draw_idle()
