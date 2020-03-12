#
# Class for quick plotting of variables from models
#
import numpy as np
import pybamm
import warnings
from collections import defaultdict


class LoopList(list):
    "A list which loops over itself when accessing an index so that it never runs out."

    def __getitem__(self, i):
        # implement looping by calling "(i) modulo (length of list)"
        return super().__getitem__(i % len(self))


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
    time_unit : str
        Format for the time output ("hours", "minutes" or "seconds")
    spatial_unit : str
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
        time_unit=None,
        spatial_unit="um",
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
        # call LoopList to make sure list index never runs out
        self.colors = LoopList(colors or ["r", "b", "k", "g", "m", "c"])
        self.linestyles = LoopList(linestyles or ["-", ":", "--", "-."])
        self.figsize = figsize or (15, 8)

        # Spatial scales (default to 1 if information not in model)
        if spatial_unit == "m":
            spatial_factor = 1
            self.spatial_unit = "m"
        elif spatial_unit == "mm":
            spatial_factor = 1e3
            self.spatial_unit = "mm"
        elif spatial_unit == "um":  # micrometers
            spatial_factor = 1e6
            self.spatial_unit = "$\mu m$"
        else:
            raise ValueError("spatial unit '{}' not recognized".format(spatial_unit))

        variables = models[0].variables
        # empty spatial scales, will raise error later if can't find a particular one
        self.spatial_scales = {}
        if "x [m]" and "x" in variables:
            x_scale = (variables["x [m]"] / variables["x"]).evaluate()[
                -1
            ] * spatial_factor
            self.spatial_scales.update(
                {
                    "negative electrode": x_scale,
                    "separator": x_scale,
                    "positive electrode": x_scale,
                }
            )
        if "y [m]" and "y" in variables:
            self.spatial_scales["current collector y"] = (
                variables["y [m]"] / variables["y"]
            ).evaluate()[-1] * spatial_factor
        if "z [m]" and "z" in variables:
            self.spatial_scales["current collector z"] = (
                variables["z [m]"] / variables["z"]
            ).evaluate()[-1] * spatial_factor
        if "r_n [m]" and "r_n" in variables:
            self.spatial_scales["negative particle"] = (
                variables["r_n [m]"] / variables["r_n"]
            ).evaluate()[-1] * spatial_factor
        if "r_p [m]" and "r_p" in variables:
            self.spatial_scales["positive particle"] = (
                variables["r_p [m]"] / variables["r_p"]
            ).evaluate()[-1] * spatial_factor

        # Time parameters
        model_timescale_in_seconds = models[0].timescale_eval
        self.ts = [solution.t for solution in solutions]
        min_t = np.min([t[0] for t in self.ts]) * model_timescale_in_seconds
        max_t = np.max([t[-1] for t in self.ts]) * model_timescale_in_seconds

        # Set timescale
        if time_unit is None:
            # defaults depend on how long the simulation is
            if max_t >= 3600:
                time_scaling_factor = 3600  # time in hours
                self.time_unit = "h"
            else:
                time_scaling_factor = 1  # time in seconds
                self.time_unit = "s"
        elif time_unit == "seconds":
            time_scaling_factor = 1
            self.time_unit = "s"
        elif time_unit == "minutes":
            time_scaling_factor = 60
            self.time_unit = "min"
        elif time_unit == "hours":
            time_scaling_factor = 3600
            self.time_unit = "h"
        else:
            raise ValueError("time unit '{}' not recognized".format(time_unit))
        self.time_scale = model_timescale_in_seconds / time_scaling_factor
        self.min_t = min_t / time_scaling_factor
        self.max_t = max_t / time_scaling_factor

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

        self.set_output_variables(output_variables, solutions)
        self.reset_axis()

    def set_output_variables(self, output_variables, solutions):
        # Set up output variables
        self.variables = {}
        self.spatial_variable_dict = {}
        self.first_dimensional_spatial_variable = {}
        self.second_dimensional_spatial_variable = {}
        self.first_spatial_scale = {}
        self.second_spatial_scale = {}

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
                    spatial_scale,
                ) = self.get_spatial_var(key, first_variable, "first")
                self.spatial_variable_dict[key] = {spatial_var_name: spatial_var_value}
                self.first_dimensional_spatial_variable[key] = (
                    spatial_var_value * spatial_scale
                )
                self.first_spatial_scale[key] = spatial_scale

            elif first_variable.dimensions == 2:
                # Don't allow 2D variables if there are multiple solutions
                if len(variables) > 1:
                    raise NotImplementedError(
                        "Cannot plot 2D variables when comparing multiple solutions, "
                        "but '{}' is 2D".format(key[0])
                    )
                # But do allow if just a single solution
                else:
                    # Add both spatial variables to the keys
                    (
                        first_spatial_var_name,
                        first_spatial_var_value,
                        first_spatial_scale,
                    ) = self.get_spatial_var(key, first_variable, "first")
                    (
                        second_spatial_var_name,
                        second_spatial_var_value,
                        second_spatial_scale,
                    ) = self.get_spatial_var(key, first_variable, "second")
                    self.spatial_variable_dict[key] = {
                        first_spatial_var_name: first_spatial_var_value,
                        second_spatial_var_name: second_spatial_var_value,
                    }
                    self.first_dimensional_spatial_variable[key] = (
                        first_spatial_var_value * first_spatial_scale
                    )
                    self.second_dimensional_spatial_variable[key] = (
                        second_spatial_var_value * second_spatial_scale
                    )

            # Store variables and subplot position
            self.variables[key] = variables
            self.subplot_positions[key] = (self.n_rows, self.n_cols, k + 1)

    def get_spatial_var(self, key, variable, dimension):
        "Return the appropriate spatial variable(s)"

        # Extract name and dimensionless value
        # Special case for current collector, which is 2D but in a weird way (both
        # first and second variables are in the same domain, not auxiliary domain)
        if dimension == "first":
            spatial_var_name = variable.first_dimension
            spatial_var_value = variable.first_dim_pts
            domain = variable.domain[0]
        elif dimension == "second":
            spatial_var_name = variable.second_dimension
            spatial_var_value = variable.second_dim_pts
            if variable.domain[0] == "current collector":
                domain = "current collector"
            else:
                domain = variable.auxiliary_domains["secondary"][0]

        if domain == "current collector":
            domain += " {}".format(spatial_var_name)

        # Get scale
        try:
            spatial_scale = self.spatial_scales[domain]
        except KeyError:
            raise KeyError(
                (
                    "Can't find spatial scale for '{}', make sure both '{} [m]' "
                    + "and '{}' are defined in the model variables"
                ).format(domain, *[spatial_var_name] * 2)
            )

        return spatial_var_name, spatial_var_value, spatial_scale

    def reset_axis(self):
        """
        Reset the axis limits to the default values.
        These are calculated to fit around the minimum and maximum values of all the
        variables in each subplot
        """
        self.axis = {}
        self.var_limits = {}
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
                x_min = self.second_dimensional_spatial_variable[key][0]
                x_max = self.second_dimensional_spatial_variable[key][-1]
                y_min = self.first_dimensional_spatial_variable[key][0]
                y_max = self.first_dimensional_spatial_variable[key][-1]

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
            else:
                self.var_limits[key] = (var_min, var_max)

    def plot(self, t):
        """Produces a quick plot with the internal states at time t.

        Parameters
        ----------
        t : float
            Dimensional time (in hours) at which to plot.
        """

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib import cm, colors

        t /= self.time_scale
        self.fig = plt.figure(figsize=self.figsize)

        self.gridspec = gridspec.GridSpec(self.n_rows, self.n_cols)
        self.plots = {}
        self.time_lines = {}
        self.axes = []

        # initialize empty handles, to be created only if the appropriate plots are made
        handles = []

        if self.n_cols == 1:
            fontsize = 30
        else:
            fontsize = 42 // self.n_cols

        for k, (key, variable_lists) in enumerate(self.variables.items()):
            ax = self.fig.add_subplot(self.gridspec[k])
            self.axes.append(ax)
            ax.set_xlim(self.axis[key][:2])
            ax.set_ylim(self.axis[key][2:])
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            self.plots[key] = defaultdict(dict)
            # Set labels for the first subplot only (avoid repetition)
            if variable_lists[0][0].dimensions == 0:
                # 0D plot: plot as a function of time, indicating time t with a line
                ax.set_xlabel("Time [{}]".format(self.time_unit), fontsize=fontsize)
                for i, variable_list in enumerate(variable_lists):
                    for j, variable in enumerate(variable_list):
                        if len(variable_list) == 1:
                            # single variable -> use linestyle to differentiate model
                            linestyle = self.linestyles[i]
                        else:
                            # multiple variables -> use linestyle to differentiate
                            # variables (color differentiates models)
                            linestyle = self.linestyles[j]
                        full_t = self.ts[i]
                        (self.plots[key][i][j],) = ax.plot(
                            full_t * self.time_scale,
                            variable(full_t, warn=False),
                            lw=2,
                            color=self.colors[i],
                            linestyle=linestyle,
                        )
                    handles.append(self.plots[key][i][0])
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
                    "{} [{}]".format(spatial_var_name, self.spatial_unit),
                    fontsize=fontsize,
                )
                # add dashed lines for boundaries between subdomains
                for bnd in variable_lists[0][0].internal_boundaries:
                    bnd_dim = bnd * self.first_spatial_scale[key]
                    y_min, y_max = self.axis[key][2:]
                    ax.plot([bnd_dim, bnd_dim], [y_min, y_max], "k--", lw=1.5)
                for i, variable_list in enumerate(variable_lists):
                    for j, variable in enumerate(variable_list):
                        if len(variable_list) == 1:
                            # single variable -> use linestyle to differentiate model
                            linestyle = self.linestyles[i]
                        else:
                            # multiple variables -> use linestyle to differentiate
                            # variables (color differentiates models)
                            linestyle = self.linestyles[j]
                        (self.plots[key][i][j],) = ax.plot(
                            self.first_dimensional_spatial_variable[key],
                            variable(t, **spatial_vars, warn=False),
                            lw=2,
                            color=self.colors[i],
                            linestyle=linestyle,
                        )
                    handles.append(self.plots[key][i][0])
            elif variable_lists[0][0].dimensions == 2:
                # 2D plot: plot as a function of x and y at time t
                # Read dictionary of spatial variables
                # note "first" and "second" variables are not in the 'right' order so
                # that, in the particle case, x is on x-axis and r is on y-axis
                # TODO: make an if statement for this
                spatial_vars = self.spatial_variable_dict[key]
                x_name = list(spatial_vars.keys())[1][0]
                y_name = list(spatial_vars.keys())[0][0]
                ax.set_xlabel(
                    "{} [{}]".format(x_name, self.spatial_unit), fontsize=fontsize
                )
                ax.set_ylabel(
                    "{} [{}]".format(y_name, self.spatial_unit), fontsize=fontsize
                )
                # there can only be one entry in the variable list
                variable = variable_lists[0][0]
                vmin, vmax = self.var_limits[key]
                ax.contourf(
                    self.second_dimensional_spatial_variable[key],
                    self.first_dimensional_spatial_variable[key],
                    variable(t, **spatial_vars, warn=False),
                    levels=100,
                    vmin=vmin,
                    vmax=vmax,
                )
                self.fig.colorbar(
                    cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax)), ax=ax
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
        if len(handles) > 0:
            self.fig.legend(handles, self.labels, loc="lower right")

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
            ax_slider = plt.axes([0.315, 0.02, 0.37, 0.03], facecolor=axcolor)
            self.slider = Slider(
                ax_slider, "Time [{}]".format(self.time_unit), 0, self.max_t, valinit=0
            )
            self.slider.on_changed(self.slider_update)

            # ignore the warning about tight layout
            warnings.simplefilter("ignore")
            bottom = 0.05 + 0.03 * max((len(self.labels) - 2), 0)
            self.gridspec.tight_layout(self.fig, rect=[0, bottom, 1, 1])
            warnings.simplefilter("always")

            if not testing:  # pragma: no cover
                plt.show()

    def slider_update(self, val):
        """
        Update the plot in self.plot() with values at new time
        """
        t = self.slider.val
        t_dimensionless = t / self.time_scale
        for k, (key, plot) in enumerate(self.plots.items()):
            if self.variables[key][0][0].dimensions == 0:
                self.time_lines[key].set_xdata([t])
            elif self.variables[key][0][0].dimensions == 1:
                for i, variable_lists in enumerate(self.variables[key]):
                    for j, variable in enumerate(variable_lists):
                        plot[i][j].set_ydata(
                            variable(
                                t_dimensionless,
                                **self.spatial_variable_dict[key],
                                warn=False
                            )
                        )
            elif self.variables[key][0][0].dimensions == 2:
                ax = self.axes[k]
                # 2D plot: plot as a function of x and y at time t
                # Read dictionary of spatial variables
                spatial_vars = self.spatial_variable_dict[key]
                # there can only be one entry in the variable list
                variable = self.variables[key][0][0]
                vmin, vmax = self.var_limits[key]
                ax.contourf(
                    self.second_dimensional_spatial_variable[key],
                    self.first_dimensional_spatial_variable[key],
                    variable(t_dimensionless, **spatial_vars, warn=False),
                    levels=100,
                    vmin=vmin,
                    vmax=vmax,
                )

        self.fig.canvas.draw_idle()
