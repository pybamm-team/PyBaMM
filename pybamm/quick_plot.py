import numpy as np
import pybamm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class QuickPlot(object):
    """
    Generates a quick plot of a subset of key outputs of the model so that the model outputs can be easily assessed.

    Parameters
    ----------
    model: :class: pybamm.BaseModel
        The model to plot the outputs of.
    mesh: :class: pybamm.Mesh
        The mesh on which the model solved
    solver: :class: pybamm.Solver
        The numerical solver for the model which contained the solution to the model.
    """

    def __init__(self, model, param, mesh, solver):

        # geometric parameters
        self.l_n = param.process_symbol(pybamm.geometric_parameters.l_n).evaluate(0, 0)
        self.l_s = param.process_symbol(pybamm.geometric_parameters.l_s).evaluate(0, 0)
        self.l_p = param.process_symbol(pybamm.geometric_parameters.l_p).evaluate(0, 0)

        # spatial and temporal variables
        self.x_n = np.linspace(0, self.l_n, 40)
        self.x_p = np.linspace(self.l_n + self.l_s, 1, 40)
        self.x = np.linspace(0, 1, 100)
        self.t = solver.t

        self.set_output_variables(solver.y, model, mesh)
        self.reset_axis()

    def set_output_variables(self, y, model, mesh):
        self.c_s_n_surf = pybamm.ProcessedVariable(
            model.variables["Negative particle surface concentration"], self.t, y, mesh
        )
        self.c_e = pybamm.ProcessedVariable(
            model.variables["Electrolyte concentration"], self.t, y, mesh
        )

        self.c_s_p_surf = pybamm.ProcessedVariable(
            model.variables["Positive particle surface concentration"], self.t, y, mesh
        )
        self.i_cell = pybamm.ProcessedVariable(
            model.variables["Total current density"], self.t, y, mesh
        )
        self.phi_s_n = pybamm.ProcessedVariable(
            model.variables["Negative electrode potential [V]"], self.t, y, mesh
        )
        self.phi_e = pybamm.ProcessedVariable(
            model.variables["Electrolyte potential [V]"], self.t, y, mesh
        )
        self.phi_s_p = pybamm.ProcessedVariable(
            model.variables["Positive electrode potential [V]"], self.t, y, mesh
        )
        self.V = pybamm.ProcessedVariable(
            model.variables["Terminal voltage [V]"], self.t, y, mesh
        )

    def reset_axis(self):
        self.axis = {
            "Negative particle surface concentration": [0, self.l_n, 0, 1],
            "Electrolyte concentration": [
                0,
                1,
                np.min(self.c_e(self.t, self.x) - 0.2),
                np.max(self.c_e(self.t, self.x) + 0.2),
            ],
            "Positive particle surface concentration": [1 - self.l_p, 1, 0, 1],
            "Total current density": [
                self.t[0],
                self.t[-1],
                np.min(self.i_cell(self.t)) - 1,
                np.max(self.i_cell(self.t)) + 1,
            ],
            "Negative electrode potential [V]": [
                0,
                self.l_n,
                np.min(self.phi_s_n(self.t, self.x_n)) - 0.01,
                np.max(self.phi_s_n(self.t, self.x_n)) + 0.01,
            ],
            "Electrolyte potential [V]": [
                0,
                1,
                np.min(self.phi_e(self.t, self.x)),
                np.max(self.phi_e(self.t, self.x)),
            ],
            "Positive electrode potential [V]": [
                1 - self.l_p,
                1,
                np.min(self.phi_s_p(self.t, self.x_p)),
                np.max(self.phi_s_p(self.t, self.x_p)),
            ],
            "Terminal voltage [V]": [
                self.t[0],
                self.t[-1],
                np.min(self.V(self.t)),
                np.max(self.V(self.t)),
            ],
        }

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

        plt.subplot(241)
        plt.xlabel("x")
        plt.ylabel("Negative particle surface concentration")
        self.negative_particle_concentration, = plt.plot(
            self.x_n, self.c_s_n_surf(t, self.x_n), lw=2
        )
        plt.axis(self.axis["Negative particle surface concentration"])

        plt.subplot(242)
        plt.xlabel("x")
        plt.ylabel("Electrolyte concentration")
        self.electrolyte_concentration, = plt.plot(self.x, self.c_e(t, self.x), lw=2)
        plt.axis(self.axis["Electrolyte concentration"])

        plt.subplot(243)
        plt.xlabel("x_p")
        plt.ylabel("Positive particle surface concentration")
        self.positive_particle_concentration, = plt.plot(
            self.x_p, self.c_s_p_surf(t, self.x_p), lw=2
        )
        plt.axis(self.axis["Positive particle surface concentration"])

        plt.subplot(244)
        plt.xlabel("Time")
        plt.ylabel("Total current density")
        self.current, = plt.plot(self.t, self.i_cell(self.t), lw=2)
        self.current_point, = plt.plot(
            [t], [self.i_cell(t)], marker="o", markersize=5, color="red"
        )
        plt.axis(self.axis["Total current density"])

        # negative electrode potential
        plt.subplot(245)
        plt.xlabel("x")
        plt.ylabel("Negative electrode potential [V]")
        self.negative_electrode_potential, = plt.plot(
            self.x_n, self.phi_s_n(t, self.x_n), lw=2
        )
        plt.axis(self.axis["Negative electrode potential [V]"])

        # electrolyte potential
        plt.subplot(246)
        plt.xlabel("x")
        plt.ylabel("Electrolyte potential [V]")
        self.electrolyte_potential, = plt.plot(self.x, self.phi_e(t, self.x), lw=2)
        plt.axis(self.axis["Electrolyte potential [V]"])

        # positive electrode potential
        plt.subplot(247)
        plt.xlabel("x")
        plt.ylabel("Positive electrode potential [V]")
        self.positive_electrode_potential, = plt.plot(
            self.x_p, self.phi_s_p(t, self.x_p), lw=2
        )
        plt.axis(self.axis["Positive electrode potential [V]"])

        # voltage
        plt.subplot(248)
        plt.xlabel("Time")
        plt.ylabel("Terminal voltage [V]")
        self.voltage, = plt.plot(self.t, self.V(self.t), lw=2)
        self.voltage_point, = plt.plot(
            [t], [self.V(t)], marker="o", markersize=5, color="red"
        )
        plt.axis(self.axis["Terminal voltage [V]"])

    def dynamic_plot(self):

        # create an initial plot at time 0
        self.plot(0)

        axcolor = "lightgoldenrodyellow"
        axfreq = plt.axes([0.315, 0.05, 0.37, 0.03], facecolor=axcolor)
        self.sfreq = Slider(axfreq, "Time", 0, self.t.max(), valinit=0)
        self.sfreq.on_changed(self.update)

        plt.subplots_adjust(
            top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5
        )

        plt.show()

    def update(self, val):
        t = self.sfreq.val
        self.negative_particle_concentration.set_ydata(self.c_s_n_surf(t, self.x_n))
        self.electrolyte_concentration.set_ydata(self.c_e(t, self.x))
        self.positive_particle_concentration.set_ydata(self.c_s_p_surf(t, self.x_p))

        self.current_point.set_data([t], [self.i_cell(t)])

        self.negative_electrode_potential.set_ydata(self.phi_s_n(t, self.x_n))
        self.electrolyte_potential.set_ydata(self.phi_e(t, self.x))
        self.positive_electrode_potential.set_ydata(self.phi_s_p(t, self.x_p))

        self.voltage_point.set_data([t], [self.V(t)])
        self.fig.canvas.draw_idle()
