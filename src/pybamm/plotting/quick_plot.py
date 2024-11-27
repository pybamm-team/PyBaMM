#
# Class for quick plotting of variables from models
#
import os
import numpy as np
import pybamm
from collections import defaultdict
from pybamm.util import import_optional_dependency


class LoopList(list):
    """A list which loops over itself when accessing an
    index so that it never runs out"""

    def __getitem__(self, i):
        # implement looping by calling "(i) modulo (length of list)"
        return super().__getitem__(i % len(self))


def ax_min(data):
    """Calculate appropriate minimum axis value for plotting"""
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    return data_max - 1.05 * (data_max - data_min)


def ax_max(data):
    """Calculate appropriate maximum axis value for plotting"""
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    return data_min + 1.05 * (data_max - data_min)


def split_long_string(title, max_words=None):
    """Get title in a nice format"""
    max_words = max_words or pybamm.settings.max_words_in_line
    words = title.split()
    # Don't split if fits on one line, don't split just for units
    if len(words) <= max_words or words[max_words].startswith("["):
        return title
    else:
        first_line = (" ").join(words[:max_words])
        second_line = (" ").join(words[max_words:])
        return first_line + "\n" + second_line


def close_plots():
    """Close all open figures"""
    plt = import_optional_dependency("matplotlib.pyplot")

    plt.close("all")


class QuickPlot:
    """
    Generates a quick plot of a subset of key outputs of the model so that the model
    outputs can be easily assessed.

    Parameters
    ----------
    solutions: (iter of) :class:`pybamm.Solution` or :class:`pybamm.Simulation`
        The numerical solution(s) for the model(s), or the simulation object(s)
        containing the solution(s).
    output_variables : list of str, optional
        List of variables to plot
    labels : list of str, optional
        Labels for the different models. Defaults to model names
    colors : list of str, optional
        The colors to loop over when plotting. Defaults to None, in which case the
        default color loop defined by matplotlib style sheet or rcParams is used.
    linestyles : list of str, optional
        The linestyles to loop over when plotting. Defaults to ["-", ":", "--", "-."]
    shading : str, optional
        The shading to use for 2D plots. Defaults to "auto".
    figsize : tuple of floats, optional
        The size of the figure to make
    n_rows : int, optional
        The number of rows to use. If None (default), floor(n // sqrt(n)) is used where
        n = len(output_variables) so that the plot is as square as possible
    time_unit : str, optional
        Format for the time output ("hours", "minutes", or "seconds")
    spatial_unit : str, optional
        Format for the spatial axes ("m", "mm", or "um")
    variable_limits : str or dict of str, optional
        How to set the axis limits (for 0D or 1D variables) or colorbar limits (for 2D
        variables). Options are:
    n_t_linear: int, optional
        The number of linearly spaced time points added to the t axis for each sub-solution.
        Note: this is only used if the solution has hermite interpolation enabled.

        - "fixed" (default): keep all axes fixes so that all data is visible
        - "tight": make axes tight to plot at each time
        - dictionary: fine-grain control for each variable, can be either "fixed" or \
        "tight" or a specific tuple (lower, upper).

    """

    def __init__(
        self,
        solutions,
        output_variables=None,
        labels=None,
        colors=None,
        linestyles=None,
        shading="auto",
        figsize=None,
        n_rows=None,
        time_unit=None,
        spatial_unit="um",
        variable_limits="fixed",
        n_t_linear=100,
    ):
        solutions = self.preprocess_solutions(solutions)

        models = [solution.all_models[0] for solution in solutions]

        # Set labels
        if labels is None:
            self.labels = [model.name for model in models]
        else:
            if len(labels) != len(models):
                raise ValueError(
                    f"labels '{labels}' have different length to models '{[model.name for model in models]}'"
                )
            self.labels = labels

        # Set colors, linestyles, figsize, axis limits
        # call LoopList to make sure list index never runs out
        if colors is None:
            self.colors = LoopList(colors or ["r", "b", "k", "g", "m", "c"])
        else:
            self.colors = LoopList(colors)
        self.linestyles = LoopList(linestyles or ["-", ":", "--", "-."])
        self.shading = shading

        # Default output variables for lead-acid and lithium-ion
        if output_variables is None:
            output_variables = models[0].default_quick_plot_variables
            # raise error if still None
            if output_variables is None:
                raise ValueError(
                    f"No default output variables provided for {models[0].name}"
                )

        # check variables have been provided after any serialisation
        if any(len(m.variables) == 0 for m in models):
            raise AttributeError("No variables to plot")

        self.n_rows = n_rows or int(
            len(output_variables) // np.sqrt(len(output_variables))
        )
        self.n_cols = int(np.ceil(len(output_variables) / self.n_rows))

        figwidth_default = min(15, 4 * self.n_cols)
        figheight_default = min(8, 1 + 3 * self.n_rows)
        self.figsize = figsize or (figwidth_default, figheight_default)

        # Spatial scales (default to 1 if information not in model)
        if spatial_unit == "m":
            self.spatial_factor = 1
            self.spatial_unit = "m"
        elif spatial_unit == "mm":
            self.spatial_factor = 1e3
            self.spatial_unit = "mm"
        elif spatial_unit == "um":  # micrometers
            self.spatial_factor = 1e6
            self.spatial_unit = r"$\mu$m"
        else:
            raise ValueError(f"spatial unit '{spatial_unit}' not recognized")

        # Time parameters
        self.ts_seconds = [solution.t for solution in solutions]
        min_t = np.min([t[0] for t in self.ts_seconds])
        max_t = np.max([t[-1] for t in self.ts_seconds])

        hermite_interp = all(sol.hermite_interpolation for sol in solutions)

        def t_sample(sol):
            if hermite_interp and n_t_linear > 2:
                # Linearly spaced time points
                t_linspace = np.linspace(sol.t[0], sol.t[-1], n_t_linear + 2)[1:-1]
                t_plot = np.union1d(sol.t, t_linspace)
            else:
                t_plot = sol.t
            return t_plot

        ts_seconds = []
        for sol in solutions:
            # Sample time points for each sub-solution
            t_sol = [t_sample(sub_sol) for sub_sol in sol.sub_solutions]
            ts_seconds.append(np.concatenate(t_sol))
        self.ts_seconds = ts_seconds

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
            raise ValueError(f"time unit '{time_unit}' not recognized")
        self.time_scaling_factor = time_scaling_factor
        self.min_t = min_t / time_scaling_factor
        self.max_t = max_t / time_scaling_factor

        # Prepare dictionary of variables
        # output_variables is a list of strings or lists, e.g.
        # ["var 1", ["variable 2", "var 3"]]
        output_variable_tuples = []
        self.variable_limits = {}
        for variable_list in output_variables:
            # Make sure we always have a list of lists of variables, e.g.
            # [["var 1"], ["variable 2", "var 3"]]
            if isinstance(variable_list, str):
                variable_list = [variable_list]

            # Store the key as a tuple
            variable_tuple = tuple(variable_list)
            output_variable_tuples.append(variable_tuple)

            # axis limits
            if variable_limits in ["fixed", "tight"]:
                self.variable_limits[variable_tuple] = variable_limits
            else:
                # If there is only one variable, extract it
                if len(variable_tuple) == 1:
                    variable = variable_tuple[0]
                else:
                    variable = variable_tuple
                try:
                    self.variable_limits[variable_tuple] = variable_limits[variable]
                except KeyError:
                    # if variable_tuple is not provided, default to "fixed"
                    self.variable_limits[variable_tuple] = "fixed"
                except TypeError as error:
                    raise TypeError(
                        "variable_limits must be 'fixed', 'tight', or a dict"
                    ) from error

        self.set_output_variables(output_variable_tuples, solutions)
        self.reset_axis()

    @staticmethod
    def preprocess_solutions(solutions):
        input_solutions = QuickPlot.check_input_validity(solutions)
        processed_solutions = []
        for sim_or_sol in input_solutions:
            if isinstance(sim_or_sol, pybamm.Simulation):
                # 'sim_or_sol' is actually a 'Simulation' object here, so it has a
                # 'Solution' attribute
                processed_solutions.append(sim_or_sol.solution)
            elif isinstance(sim_or_sol, pybamm.Solution):
                processed_solutions.append(sim_or_sol)
        return processed_solutions

    @staticmethod
    def check_input_validity(input_solutions):
        if not isinstance(input_solutions, (pybamm.Solution, pybamm.Simulation, list)):
            raise TypeError(
                "Solutions must be 'pybamm.Solution' or 'pybamm.Simulation' or list"
            )
        elif not isinstance(input_solutions, list):
            input_solutions = [input_solutions]
        else:
            if not input_solutions:
                raise TypeError("QuickPlot requires at least 1 solution or simulation.")
        return input_solutions

    def set_output_variables(self, output_variables, solutions):
        # Set up output variables
        self.variables = {}
        self.spatial_variable_dict = {}
        self.first_spatial_variable = {}
        self.second_spatial_variable = {}
        self.x_first_and_y_second = {}
        self.is_y_z = {}

        # Calculate subplot positions based on number of variables supplied
        self.subplot_positions = {}

        for k, variable_tuple in enumerate(output_variables):
            # Prepare list of variables
            variables = [None] * len(solutions)

            # process each variable in variable_list for each model
            for i, solution in enumerate(solutions):
                # variables lists of lists, so variables[i] is a list
                variables[i] = []
                for var in variable_tuple:
                    sol = solution[var]
                    # Check variable isn't all-nan
                    if np.all(np.isnan(sol.entries)):
                        raise ValueError(f"All-NaN variable '{var}' provided")
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
                        f"'{variable_tuple[0]}' has domain '{domain}', but '{variable_tuple[idx]}' has domain '{variable.domain}'"
                    )
                self.spatial_variable_dict[variable_tuple] = {}

            # Set the x variable (i.e. "x" or "r" for any one-dimensional variables)
            if first_variable.dimensions == 1:
                (spatial_var_name, spatial_var_value) = self._get_spatial_var(
                    variable_tuple, first_variable, "first"
                )
                self.spatial_variable_dict[variable_tuple] = {
                    spatial_var_name: spatial_var_value
                }
                self.first_spatial_variable[variable_tuple] = (
                    spatial_var_value * self.spatial_factor
                )

            elif first_variable.dimensions == 2:
                # Don't allow 2D variables if there are multiple solutions
                if len(variables) > 1:
                    raise NotImplementedError(
                        "Cannot plot 2D variables when comparing multiple solutions, "
                        f"but '{variable_tuple[0]}' is 2D"
                    )
                # But do allow if just a single solution
                else:
                    # Add both spatial variables to the variable_tuples
                    (
                        first_spatial_var_name,
                        first_spatial_var_value,
                    ) = self._get_spatial_var(variable_tuple, first_variable, "first")
                    (
                        second_spatial_var_name,
                        second_spatial_var_value,
                    ) = self._get_spatial_var(variable_tuple, first_variable, "second")
                    self.spatial_variable_dict[variable_tuple] = {
                        first_spatial_var_name: first_spatial_var_value,
                        second_spatial_var_name: second_spatial_var_value,
                    }
                    self.first_spatial_variable[variable_tuple] = (
                        first_spatial_var_value * self.spatial_factor
                    )
                    self.second_spatial_variable[variable_tuple] = (
                        second_spatial_var_value * self.spatial_factor
                    )
                    # different order based on whether the domains
                    # are x-r, x-z or y-z, etc
                    if (
                        first_spatial_var_name in ("r", "R")
                        and second_spatial_var_name == "x"
                    ):
                        self.x_first_and_y_second[variable_tuple] = False
                        self.is_y_z[variable_tuple] = False
                    elif (
                        first_spatial_var_name == "y" and second_spatial_var_name == "z"
                    ):
                        self.x_first_and_y_second[variable_tuple] = True
                        self.is_y_z[variable_tuple] = True
                    else:
                        self.x_first_and_y_second[variable_tuple] = True
                        self.is_y_z[variable_tuple] = False

            # Store variables and subplot position
            self.variables[variable_tuple] = variables
            self.subplot_positions[variable_tuple] = (self.n_rows, self.n_cols, k + 1)

    def _get_spatial_var(self, key, variable, dimension):
        """Return the appropriate spatial variable(s)"""

        # Extract name and value
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
                domain = variable.domains["secondary"][0]

        if domain == "current collector":
            domain += f" {spatial_var_name}"

        return spatial_var_name, spatial_var_value

    def reset_axis(self):
        """
        Reset the axis limits to the default values.
        These are calculated to fit around the minimum and maximum values of all the
        variables in each subplot
        """
        self.axis_limits = {}
        for key, variable_lists in self.variables.items():
            if variable_lists[0][0].dimensions == 0:
                x_min = self.min_t
                x_max = self.max_t
            elif variable_lists[0][0].dimensions == 1:
                x_min = self.first_spatial_variable[key][0]
                x_max = self.first_spatial_variable[key][-1]
            elif variable_lists[0][0].dimensions == 2:
                # different order based on whether the domains are x-r, x-z or y-z, etc
                if self.x_first_and_y_second[key] is False:
                    x_min = self.second_spatial_variable[key][0]
                    x_max = self.second_spatial_variable[key][-1]
                    y_min = self.first_spatial_variable[key][0]
                    y_max = self.first_spatial_variable[key][-1]
                else:
                    x_min = self.first_spatial_variable[key][0]
                    x_max = self.first_spatial_variable[key][-1]
                    y_min = self.second_spatial_variable[key][0]
                    y_max = self.second_spatial_variable[key][-1]

                # Create axis for contour plot
                self.axis_limits[key] = [x_min, x_max, y_min, y_max]

            # Get min and max variable values
            if self.variable_limits[key] == "fixed":
                # fixed variable limits: calculate "globlal" min and max
                spatial_vars = self.spatial_variable_dict[key]
                var_min = np.min(
                    [
                        ax_min(var(self.ts_seconds[i], **spatial_vars))
                        for i, variable_list in enumerate(variable_lists)
                        for var in variable_list
                    ]
                )
                var_max = np.max(
                    [
                        ax_max(var(self.ts_seconds[i], **spatial_vars))
                        for i, variable_list in enumerate(variable_lists)
                        for var in variable_list
                    ]
                )
                if np.isnan(var_min) or np.isnan(var_max):
                    raise ValueError(
                        "The variable limits are set to 'fixed' but the min and max "
                        "values are NaN"
                    )
                if var_min == var_max:
                    var_min -= 1
                    var_max += 1
            elif self.variable_limits[key] == "tight":
                # tight variable limits: axes will adjust each time
                var_min, var_max = None, None
            else:
                # user-specified axis limits
                var_min, var_max = self.variable_limits[key]

            if variable_lists[0][0].dimensions in [0, 1]:
                self.axis_limits[key] = [x_min, x_max, var_min, var_max]
            else:
                self.variable_limits[key] = (var_min, var_max)
            if (
                var_min is not None
                and var_max is not None
                and (np.isnan(var_min) or np.isnan(var_max))
            ):  # pragma: no cover
                raise ValueError(f"Axis limits cannot be NaN for variables '{key}'")

    def plot(self, t, dynamic=False):
        """Produces a quick plot with the internal states at time t.

        Parameters
        ----------
        t : float
            Dimensional time (in 'time_units') at which to plot.
        dynamic : bool, optional
            Determine whether to allocate space for a slider at the bottom of the plot when generating a dynamic plot.
            If True, creates a dynamic plot with a slider.
        """

        plt = import_optional_dependency("matplotlib.pyplot")
        gridspec = import_optional_dependency("matplotlib.gridspec")
        cm = import_optional_dependency("matplotlib", "cm")
        colors = import_optional_dependency("matplotlib", "colors")

        t_in_seconds = t * self.time_scaling_factor
        self.fig = plt.figure(figsize=self.figsize)

        self.gridspec = gridspec.GridSpec(self.n_rows, self.n_cols)
        self.plots = {}
        self.time_lines = {}
        self.colorbars = {}
        self.axes = QuickPlotAxes()

        # initialize empty handles, to be created only if the appropriate plots are made
        solution_handles = []

        for k, (key, variable_lists) in enumerate(self.variables.items()):
            ax = self.fig.add_subplot(self.gridspec[k])
            self.axes.add(key, ax)
            x_min, x_max, y_min, y_max = self.axis_limits[key]
            ax.set_xlim(x_min, x_max)
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            self.plots[key] = defaultdict(dict)
            variable_handles = []
            # Set labels for the first subplot only (avoid repetition)
            if variable_lists[0][0].dimensions == 0:
                # 0D plot: plot as a function of time, indicating time t with a line
                ax.set_xlabel(f"Time [{self.time_unit}]")
                for i, variable_list in enumerate(variable_lists):
                    for j, variable in enumerate(variable_list):
                        if len(variable_list) == 1:
                            # single variable -> use linestyle to differentiate model
                            linestyle = self.linestyles[i]
                        else:
                            # multiple variables -> use linestyle to differentiate
                            # variables (color differentiates models)
                            linestyle = self.linestyles[j]
                        full_t = self.ts_seconds[i]
                        (self.plots[key][i][j],) = ax.plot(
                            full_t / self.time_scaling_factor,
                            variable(full_t),
                            color=self.colors[i],
                            linestyle=linestyle,
                        )
                        variable_handles.append(self.plots[key][0][j])
                    solution_handles.append(self.plots[key][i][0])
                y_min, y_max = ax.get_ylim()
                ax.set_ylim(y_min, y_max)
                (self.time_lines[key],) = ax.plot(
                    [
                        t_in_seconds / self.time_scaling_factor,
                        t_in_seconds / self.time_scaling_factor,
                    ],
                    [y_min, y_max],
                    "k--",
                    lw=1.5,
                )
            elif variable_lists[0][0].dimensions == 1:
                # 1D plot: plot as a function of x at time t
                # Read dictionary of spatial variables
                spatial_vars = self.spatial_variable_dict[key]
                spatial_var_name = next(iter(spatial_vars.keys()))
                ax.set_xlabel(
                    f"{spatial_var_name} [{self.spatial_unit}]",
                )
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
                            self.first_spatial_variable[key],
                            variable(t_in_seconds, **spatial_vars),
                            color=self.colors[i],
                            linestyle=linestyle,
                            zorder=10,
                        )
                        variable_handles.append(self.plots[key][0][j])
                    solution_handles.append(self.plots[key][i][0])
                # add lines for boundaries between subdomains
                for boundary in variable_lists[0][0].internal_boundaries:
                    boundary_scaled = boundary * self.spatial_factor
                    ax.axvline(boundary_scaled, color="0.5", lw=1, zorder=0)
            elif variable_lists[0][0].dimensions == 2:
                # Read dictionary of spatial variables
                spatial_vars = self.spatial_variable_dict[key]
                # there can only be one entry in the variable list
                variable = variable_lists[0][0]
                # different order based on whether the domains are x-r, x-z or y-z, etc
                if self.x_first_and_y_second[key] is False:
                    x_name = list(spatial_vars.keys())[1][0]
                    y_name = next(iter(spatial_vars.keys()))[0]
                    x = self.second_spatial_variable[key]
                    y = self.first_spatial_variable[key]
                    var = variable(t_in_seconds, **spatial_vars)
                else:
                    x_name = next(iter(spatial_vars.keys()))[0]
                    y_name = list(spatial_vars.keys())[1][0]
                    x = self.first_spatial_variable[key]
                    y = self.second_spatial_variable[key]
                    var = variable(t_in_seconds, **spatial_vars).T
                ax.set_xlabel(f"{x_name} [{self.spatial_unit}]")
                ax.set_ylabel(f"{y_name} [{self.spatial_unit}]")
                vmin, vmax = self.variable_limits[key]
                # store the plot and the var data (for testing) as cant access
                # z data from QuadMesh or QuadContourSet object
                if self.is_y_z[key] is True:
                    self.plots[key][0][0] = ax.pcolormesh(
                        x,
                        y,
                        var,
                        vmin=vmin,
                        vmax=vmax,
                        shading=self.shading,
                    )
                else:
                    self.plots[key][0][0] = ax.contourf(
                        x, y, var, levels=100, vmin=vmin, vmax=vmax
                    )
                self.plots[key][0][1] = var
                if vmin is None and vmax is None:
                    vmin = ax_min(var)
                    vmax = ax_max(var)
                self.colorbars[key] = self.fig.colorbar(
                    cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax)),
                    ax=ax,
                )
            # Set either y label or legend entries
            if len(key) == 1:
                title = split_long_string(key[0])
                ax.set_title(title, fontsize="medium")
            else:
                ax.legend(
                    variable_handles,
                    [split_long_string(s, 6) for s in key],
                    bbox_to_anchor=(0.5, 1),
                    loc="lower center",
                )

        # Set global legend
        if len(self.labels) > 1:
            fig_legend = self.fig.legend(
                solution_handles, self.labels, loc="lower right"
            )
            # Get the position of the top of the legend in relative figure units
            # There may be a better way ...
            try:
                legend_top_inches = fig_legend.get_window_extent(
                    renderer=self.fig.canvas.get_renderer()
                ).get_points()[1, 1]
                fig_height_inches = (self.fig.get_size_inches() * self.fig.dpi)[1]
                legend_top = legend_top_inches / fig_height_inches
            except AttributeError:  # pragma: no cover
                # When testing the examples we set the matplotlib backend to "Template"
                # which means that the above code doesn't work. Since this is just for
                # that particular test we can just skip it
                legend_top = 0
        else:
            legend_top = 0

        # Fix layout
        if dynamic:
            slider_top = 0.05
        else:
            slider_top = 0
        bottom = max(legend_top, slider_top)
        self.gridspec.tight_layout(self.fig, rect=[0, bottom, 1, 1])

    def dynamic_plot(self, show_plot=True, step=None):
        """
        Generate a dynamic plot with a slider to control the time.

        Parameters
        ----------
        step : float, optional
            For notebook mode, size of steps to allow in the slider. Defaults to 1/100th
            of the total time.
        show_plot : bool, optional
            Whether to show the plots. Default is True. Set to False if you want to
            only display the plot after plt.show() has been called.

        """
        if pybamm.is_notebook():  # pragma: no cover
            import ipywidgets as widgets

            step = step or self.max_t / 100
            widgets.interact(
                lambda t: self.plot(t, dynamic=False),
                t=widgets.FloatSlider(
                    min=self.min_t, max=self.max_t, step=step, value=self.min_t
                ),
                continuous_update=False,
            )
        else:
            plt = import_optional_dependency("matplotlib.pyplot")
            Slider = import_optional_dependency("matplotlib.widgets", "Slider")

            # create an initial plot at time self.min_t
            self.plot(self.min_t, dynamic=True)

            axcolor = "lightgoldenrodyellow"
            ax_slider = plt.axes([0.315, 0.02, 0.37, 0.03], facecolor=axcolor)
            self.slider = Slider(
                ax_slider,
                f"Time [{self.time_unit}]",
                self.min_t,
                self.max_t,
                valinit=self.min_t,
                color="#1f77b4",
            )
            self.slider.on_changed(self.slider_update)

            if show_plot:  # pragma: no cover
                plt.show()

    def slider_update(self, t):
        """
        Update the plot in self.plot() with values at new time
        """
        from matplotlib import cm, colors

        time_in_seconds = t * self.time_scaling_factor
        for k, (key, plot) in enumerate(self.plots.items()):
            ax = self.axes[k]
            if self.variables[key][0][0].dimensions == 0:
                self.time_lines[key].set_xdata([t])
            elif self.variables[key][0][0].dimensions == 1:
                var_min = np.inf
                var_max = -np.inf
                for i, variable_lists in enumerate(self.variables[key]):
                    for j, variable in enumerate(variable_lists):
                        var = variable(
                            time_in_seconds,
                            **self.spatial_variable_dict[key],
                        )
                        plot[i][j].set_ydata(var)
                        var_min = min(var_min, ax_min(var))
                        var_max = max(var_max, ax_max(var))
                # update boundaries between subdomains
                y_min, y_max = self.axis_limits[key][2:]
                if y_min is None and y_max is None:
                    ax.set_ylim(var_min, var_max)
            elif self.variables[key][0][0].dimensions == 2:
                # 2D plot: plot as a function of x and y at time t
                # Read dictionary of spatial variables
                spatial_vars = self.spatial_variable_dict[key]
                # there can only be one entry in the variable list
                variable = self.variables[key][0][0]
                vmin, vmax = self.variable_limits[key]
                if self.x_first_and_y_second[key] is False:
                    x = self.second_spatial_variable[key]
                    y = self.first_spatial_variable[key]
                    var = variable(time_in_seconds, **spatial_vars)
                else:
                    x = self.first_spatial_variable[key]
                    y = self.second_spatial_variable[key]
                    var = variable(time_in_seconds, **spatial_vars).T
                # store the plot and the var data (for testing) as cant access
                # z data from QuadMesh or QuadContourSet object
                if self.is_y_z[key] is True:
                    self.plots[key][0][0] = ax.pcolormesh(
                        x,
                        y,
                        var,
                        vmin=vmin,
                        vmax=vmax,
                        shading=self.shading,
                    )
                else:
                    self.plots[key][0][0] = ax.contourf(
                        x, y, var, levels=100, vmin=vmin, vmax=vmax
                    )
                self.plots[key][0][1] = var
                if (vmin, vmax) == (None, None):
                    vmin = ax_min(var)
                    vmax = ax_max(var)
                    cb = self.colorbars[key]
                    cb.update_normal(
                        cm.ScalarMappable(colors.Normalize(vmin=vmin, vmax=vmax))
                    )

        self.fig.canvas.draw_idle()

    def create_gif(self, number_of_images=80, duration=0.1, output_filename="plot.gif"):
        """
        Generates x plots over a time span of max_t - min_t and compiles them to create
        a GIF.

        Parameters
        ----------
        number_of_images : int, optional
            Number of images/plots to be compiled for a GIF.
        duration : float, optional
            Duration of visibility of a single image/plot in the created GIF.
        output_filename : str, optional
            Name of the generated GIF file.

        """
        imageio = import_optional_dependency("imageio.v2")
        plt = import_optional_dependency("matplotlib.pyplot")

        # time stamps at which the images/plots will be created
        time_array = np.linspace(self.min_t, self.max_t, num=number_of_images)
        images = []

        # create images/plots
        stub_name = output_filename.split(".")[0]
        for val in time_array:
            self.plot(val)
            temp_name = f"{stub_name}{val}.png"
            images.append(temp_name)
            self.fig.savefig(temp_name, dpi=300)
            plt.close()

        # compile the images/plots to create a GIF
        with imageio.get_writer(output_filename, mode="I", duration=duration) as writer:
            for image in images:
                writer.append_data(imageio.imread(image))

        # remove the generated images
        for image in images:
            os.remove(image)


class QuickPlotAxes:
    """
    Class to store axes for the QuickPlot
    """

    def __init__(self):
        self._by_variable = {}
        self._axes = []

    def add(self, keys, axis):
        """
        Add axis

        Parameters
        ----------
        keys : iter
            Iterable of keys of variables being plotted on the axis
        axis : matplotlib Axis object
            The axis object
        """
        self._axes.append(axis)
        for k in keys:
            self._by_variable[k] = axis

    def __getitem__(self, index):
        """
        Get axis by index
        """
        return self._axes[index]

    def by_variable(self, key):
        """
        Get axis by variable name
        """
        return self._by_variable[key]
