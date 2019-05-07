import numpy as np
import pybamm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def quick_plot(model, param, mesh, solver):
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

    # obtain parameters values

    l_n = param.process_symbol(pybamm.geometric_parameters.l_n).evaluate(0, 0)
    l_s = param.process_symbol(pybamm.geometric_parameters.l_s).evaluate(0, 0)
    l_p = param.process_symbol(pybamm.geometric_parameters.l_p).evaluate(0, 0)

    # TODO: change mesh interface at somepoint (cannot just )
    # unpack the mesh to obtain discrete spatial variables

    x_n = np.linspace(0, l_n, 40)
    x_s = np.linspace(l_n, l_n + l_s, 20)
    x_p = np.linspace(l_n + l_s, 1, 40)
    x = np.linspace(0, 1, 100)

    r_n = np.linspace(0, 1, 100)
    r_p = np.linspace(0, 1, 100)

    all_time = solver.t

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # electrolyte concentration

    c_e = pybamm.ProcessedVariable(
        model.variables["Electrolyte concentration"], solver.t, solver.y, mesh=mesh
    )

    plt.subplot(342)
    plt.title("Electrolyte Concentration")
    plt.xlabel("x")
    plt.ylabel("c_e")
    electrolyte_concentration, = plt.plot(x, c_e(t=0, x=x), lw=2)
    plt.axis([0, 1, np.min(c_e(all_time, x)), np.max(c_e(all_time, x))])

    # plt.subplot(342)
    # plt.title("Negative Electrode")
    # plt.xlabel("x")
    # plt.ylabel("Lithium concentration")
    # negCon, = plt.plot(sol.grid.xn, sol.cn[:, 1, 0], lw=2)
    # plt.axis([0, sol.grid.xn[-1], 0, 1])

    # plt.subplot(343)
    # plt.title("Positive Electrode")
    # plt.xlabel("x")
    # plt.ylabel("Lithium concentration")
    # posCon, = plt.plot(sol.grid.xp, sol.cn[:, 1, 0], lw=2)
    # plt.axis([sol.grid.xp[0], sol.grid.xp[-1], 0, 1])

    # plt.subplot(344)
    # plt.xlabel("Time")
    # plt.ylabel("Current")
    # curr, = plt.plot(sol.time, sol.current * np.ones(np.shape(sol.time)), lw=2)
    # plt.hold(True)
    # curr_point, = plt.plot(
    #     [sol.time[0]], [sol.current], marker="o", markersize=5, color="red"
    # )

    # plt.subplot(345)
    # plt.xlabel("x")
    # plt.ylabel("Electrolyte Potential")
    # Psi, = plt.plot(
    #     sol.grid.x,
    #     np.concatenate((sol.Psi_n[:, 0], sol.Psi_s[:, 0], sol.Psi_p[:, 0])),
    #     lw=2,
    # )
    # plt.axis([0, 1, np.min(sol.Psi_n), np.max(sol.Psi_p)])

    # plt.subplot(346)
    # plt.xlabel("x")
    # plt.ylabel("Electrode Potential")
    # psi_n, = plt.plot(sol.grid.xn, sol.psi_n[:, 0], lw=2)
    # plt.axis([sol.grid.xn[0], sol.grid.xn[-1], -1, 1])

    # plt.subplot(347)
    # plt.xlabel("x")
    # plt.ylabel("Electrode Potential")
    # psi_p, = plt.plot(sol.grid.xp, sol.psi_p[:, 0], lw=2)
    # plt.axis([sol.grid.xp[0], sol.grid.xp[-1], 0, 5])

    # plt.subplot(348)
    # plt.xlabel("Time")
    # plt.ylabel("Voltage")
    # voltage, = plt.plot(sol.time, sol.voltage, lw=2)
    # plt.hold(True)
    # voltage_point, = plt.plot(
    #     [sol.time[0]], [sol.voltage[0]], marker="o", markersize=5, color="red"
    # )
    # # plt.axis([sol.time[0], sol.time[-1], -1, 1])

    # plt.subplot(349)
    # plt.xlabel("x")
    # plt.ylabel("Electrolyte Current")
    # J_sol = np.concatenate(
    #     (sol.Jn[:, 0], sol.current * np.ones((sol.grid.sep - 1)), sol.Jp[:, 0])
    # )
    # J, = plt.plot(sol.grid.x_edge, J_sol, lw=2)
    # # plt.axis([sol.grid.x[0], sol.grid.x[-1], 0, 1])

    # plt.subplot(3, 4, 10)
    # plt.xlabel("x")
    # plt.ylabel("Electrode Current")
    # jn, = plt.plot(sol.grid.xn_edge, sol.jn[:, 0], lw=2)

    # plt.subplot(3, 4, 11)
    # plt.xlabel("x")
    # plt.ylabel("Electrode Current")
    # jp, = plt.plot(sol.grid.xp_edge, sol.jp[:, 0], lw=2)

    # plt.subplot(3, 4, 12)
    # plt.xlabel("Time")
    # plt.ylabel("Normalised Capacity")
    # plt.plot(sol.time, sol.capacity, lw=2)
    # plt.hold(True)
    # cap_point, = plt.plot(
    #     [sol.time[0]], [sol.capacity[0]], marker="o", markersize=5, color="red"
    # )
    # plt.axis([sol.time[0], sol.time[-1], 0.9, 1.1])

    axcolor = "lightgoldenrodyellow"
    axfreq = plt.axes([0.315, 0.05, 0.37, 0.03], facecolor=axcolor)
    sfreq = Slider(axfreq, "Time", 0, all_time.max(), valinit=0)

    def update(val):
        # t = int(round(sfreq.val))
        time = sfreq.val
        electrolyte_concentration.set_ydata(c_e(t=time, x=x))
        # negCon.set_ydata(sol.cn[:, -1, t])
        # posCon.set_ydata(sol.cp[:, -1, t])
        # Psi.set_ydata(
        #     np.concatenate((sol.Psi_n[:, t], sol.Psi_s[:, t], sol.Psi_p[:, t]))
        # )
        # psi_n.set_ydata(sol.psi_n[:, t])
        # psi_p.set_ydata(sol.psi_p[:, t])

        # voltage_point.set_data([sol.time[t]], [sol.voltage[t]])
        # curr_point.set_data([sol.time[t]], [sol.current])
        # cap_point.set_data([sol.time[t]], [sol.capacity[t]])

        # J_sol = np.concatenate(
        #     (sol.Jn[:, t], sol.current * np.ones((sol.grid.sep - 1)), sol.Jp[:, t])
        # )
        # J.set_ydata(J_sol)

        # jn.set_ydata(sol.jn[:, t])
        # jp.set_ydata(sol.jp[:, t])

        fig.canvas.draw_idle()

    sfreq.on_changed(update)

    plt.subplots_adjust(
        top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5
    )

    plt.show()

