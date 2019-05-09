import numpy as np
import pybamm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


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

    def __init__(self, models, param, mesh, solvers):
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

        # geometric parameters
        self.l_n = param.process_symbol(pybamm.geometric_parameters.l_n).evaluate(0, 0)
        self.l_s = param.process_symbol(pybamm.geometric_parameters.l_s).evaluate(0, 0)
        self.l_p = param.process_symbol(pybamm.geometric_parameters.l_p).evaluate(0, 0)

        # spatial and temporal variables
        self.x_n = np.linspace(0, self.l_n, 40)
        self.x_p = np.linspace(self.l_n + self.l_s, 1, 40)
        self.x = np.linspace(0, 1, 100)
        self.ts = [solver.t for solver in solvers]
        self.max_t = np.max([t[-1] for t in self.ts])

        output_variables = [
            "Negative particle surface concentration",
            "Electrolyte concentration",
            "Positive particle surface concentration",
            "Negative electrode potential [V]",
            "Electrolyte potential [V]",
            "Positive electrode potential [V]",
        ]
        # "Total current density",
        # "Terminal voltage [V]",
        self.set_output_variables(output_variables, solvers, models, mesh)
        self.reset_axis()

    def set_output_variables(self, output_variables, solvers, models, mesh):
        # Set up output variables
        self.variables = {}
        self.x_values = {}
        for var in output_variables:
            self.variables[var] = [
                pybamm.ProcessedVariable(
                    models[i].variables[var], solvers[i].t, solvers[i].y, mesh
                )
                for i in range(len(models))
            ]
            domain = models[0].variables[var].domain
            self.x_values[var] = mesh.combine_submeshes(*domain)[0].edges

    def reset_axis(self):
        """
        Reset the axis limits to the default values.
        """
        self.axis = {}
        for name, variable in self.variables.items():
            x = self.x_values[name]
            y_min = np.min(
                [np.min(v(self.ts[i], x) - 0.2) for i, v in enumerate(variable)]
            )
            y_max = np.max(
                [np.max(v(self.ts[i], x) + 0.2) for i, v in enumerate(variable)]
            )
            self.axis[name] = [x[0], x[-1], y_min, y_max]

        # # "Total current density": [
        # #     self.ts[0][0],
        # #     self.max_t,
        # #     np.min(
        # #         [
        # #             np.min(v(self.ts[i], self.x) - 1)
        # #             for i, v in enumerate(self.variables["Total current density"])
        # #         ]
        # #     ),
        # #     np.max(
        # #         [
        # #             np.max(v(self.ts[i], self.x) + 1)
        # #             for i, v in enumerate(self.variables["Total current density"])
        # #         ]
        # #     ),
        # # ],
        # "Terminal voltage [V]": [
        #     self.ts[0][0],
        #     self.max_t,
        #     np.min(
        #         [
        #             np.min(v(self.ts[i], self.x))
        #             for i, v in enumerate(self.variables["Terminal voltage [V]"])
        #         ]
        #     ),
        #     np.max(
        #         [
        #             np.max(v(self.ts[i], self.x))
        #             for i, v in enumerate(self.variables["Terminal voltage [V]"])
        #         ]
        #     ),
        # ],

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

        for k, name in enumerate(self.variables.keys()):
            variable = self.variables[name]
            plt.subplot(2, 4, k + 1)
            plt.xlabel("x")
            plt.ylabel(name)
            self.plots[name] = [None] * self.num_models
            x_value = self.x  # _values[name]
            for i in range(self.num_models):
                self.plots[name][i], = plt.plot(x_value, variable[i](t, x_value), lw=2)
            plt.axis(self.axis[name])

        # plt.subplot(244)
        # plt.xlabel("Time")
        # plt.ylabel("Total current density")
        # self.current, = plt.plot(self.ts[0], self.i_cell[i](self.ts[0]), lw=2)
        # self.current_point, = plt.plot(
        #     [t], [self.i_cell[i](t)], marker="o", markersize=5, color="red"
        # )
        # plt.axis(self.axis["Total current density"])
        #
        # # voltage
        # plt.subplot(248)
        # plt.xlabel("Time")
        # plt.ylabel("Terminal voltage [V]")
        # self.voltage, = plt.plot(self.ts[0], self.V[i](self.ts[0]), lw=2)
        # self.voltage_point, = plt.plot(
        #     [t], [self.V[i](t)], marker="o", markersize=5, color="red"
        # )
        # plt.axis(self.axis["Terminal voltage [V]"])

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
            for i in range(self.num_models):
                x_value = self.x  # _values[var]
                plot[i].set_ydata(self.variables[var][i](t, x_value))

                # self.current_point.set_data([t], [self.i_cell[i](t)])
                # self.voltage_point.set_data([t], [self.V[i](t)])
        self.fig.canvas.draw_idle()
