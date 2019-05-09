import numpy as np
import pybamm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


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
    solvers: (iter of) :class:`pybamm.Solver`
        The numerical solver(s) for the model(s) which contained the solution to the
        model(s).
    output_variables : list of str
        List of variables to plot
    """

    def __init__(self, models, mesh, solvers, output_variables=None):
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

        # Scales
        vars = models[0].variables
        self.x_scale = (vars["x [m]"] / vars["x"]).evaluate()[-1]
        self.time_scale = (vars["Time [h]"] / vars["Time"]).evaluate(t=1)

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
                    "Interfacial current density [A m-2]",
                    "Electrolyte concentration [mols m-3]",
                    "Current [A]",
                    "Porosity",
                    "Electrolyte potential [V]",
                    "Terminal voltage [V]",
                ]
        self.set_output_variables(output_variables, solvers, models, mesh)
        self.reset_axis()

    def set_output_variables(self, output_variables, solvers, models, mesh):
        # Set up output variables
        self.variables = {}
        self.x_values = {}

        # Calculate subplot positions based on number of variables supplied
        self.subplot_positions = {}
        n = int(len(output_variables) // np.sqrt(len(output_variables)))
        m = np.ceil(len(output_variables) / n)

        # Process output variables into a form that can be plotted
        for k, var in enumerate(output_variables):
            self.variables[var] = [
                pybamm.ProcessedVariable(
                    models[i].variables[var], solvers[i].t, solvers[i].y, mesh
                )
                for i in range(len(models))
            ]
            if self.variables[var][0].dimensions == 2:
                domain = models[0].variables[var].domain
                self.x_values[var] = mesh.combine_submeshes(*domain)[0].edges
            self.subplot_positions[var] = (n, m, k + 1)

        # Set labels
        self.labels = [model.name for model in models]

        # Don't allow 3D variables
        if self.variables[var][0].dimensions == 3:
            raise NotImplementedError("cannot plot 3D variables")

    def reset_axis(self):
        """
        Reset the axis limits to the default values.
        """
        self.axis = {}
        for name, variable in self.variables.items():
            if variable[0].dimensions == 1:
                y_min = np.min([ax_min(v(self.ts[i])) for i, v in enumerate(variable)])
                y_max = np.max([ax_max(v(self.ts[i])) for i, v in enumerate(variable)])
                self.axis[name] = [
                    self.min_t * self.time_scale,
                    self.max_t * self.time_scale,
                    y_min,
                    y_max,
                ]
            elif variable[0].dimensions == 2:
                x = self.x_values[name]
                x_scaled = x * self.x_scale
                y_min = np.min(
                    [ax_min(v(self.ts[i], x)) for i, v in enumerate(variable)]
                )
                y_max = np.max(
                    [ax_max(v(self.ts[i], x)) for i, v in enumerate(variable)]
                )
                self.axis[name] = [x_scaled[0], x_scaled[-1], y_min, y_max]

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

        for k, name in enumerate(self.variables.keys()):
            variable = self.variables[name]
            plt.subplot(*self.subplot_positions[name])
            plt.ylabel(name, fontsize=14)
            plt.axis(self.axis[name])
            self.plots[name] = [None] * self.num_models
            # Set labels
            if k == 0:
                labels = self.labels
            else:
                labels = [None] * len(self.labels)
            if variable[0].dimensions == 2:
                # 2D plot: plot as a function of x at time t
                plt.xlabel("Position [m]", fontsize=14)
                x_value = self.x_values[name]
                for i in range(self.num_models):
                    self.plots[name][i], = plt.plot(
                        x_value * self.x_scale,
                        variable[i](t, x_value),
                        lw=2,
                        label=labels[i],
                    )
            else:
                # 1D plot: plot as a function of time, indicating time t with a line
                plt.xlabel("Time [h]", fontsize=14)
                for i in range(self.num_models):
                    full_t = self.ts[i]
                    self.plots[name][i], = plt.plot(
                        full_t * self.time_scale,
                        variable[i](full_t),
                        lw=2,
                        label=labels[i],
                    )
                    y_min, y_max = self.axis[name][2:]
                    self.time_lines[name], = plt.plot([t, t], [y_min, y_max], "k--")
        self.fig.legend(loc="lower right")

    def dynamic_plot(self, testing=False):
        """
        Generate a dynamic plot with a slider to control the time. We recommend using
        ipywidgets instead of this function if you are using jupyter notebooks
        """

        # create an initial plot at time 0
        self.plot(0)

        axcolor = "lightgoldenrodyellow"
        axfreq = plt.axes([0.315, 0.02, 0.37, 0.03], facecolor=axcolor)
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
            if self.variables[var][0].dimensions == 2:
                x = self.x_values[var]
                for i in range(self.num_models):
                    plot[i].set_ydata(self.variables[var][i](t, x))
            else:
                self.time_lines[var].set_xdata([t])

        self.fig.canvas.draw_idle()
