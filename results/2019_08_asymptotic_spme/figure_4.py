#
# Figure 4: DFN, SPMe internal states comparison
#
import pybamm
import numpy as np
import matplotlib.pyplot as plt

generate_plots = True
print_RMSE = True
export_data = False

models = [pybamm.lithium_ion.SPMe(), pybamm.lithium_ion.DFN()]

param = models[0].default_parameter_values
for model in models:
    param.process_model(model)

var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 20, var.x_p: 30, var.r_n: 15, var.r_p: 15}
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5}

discs = []
for model in models:
    # create geometry
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, models[-1].default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    discs.append(disc)

C_rates = [0.1, 0.5, 1, 2, 3]
colour = {0.1: "r", 0.5: "b", 1: "g", 2: "m", 3: "y"}

negative_variables = [
    "Negative particle surface concentration",
    "Negative electrode potential [V]",
]
positive_variables = [
    "Positive particle surface concentration",
    "Positive electrode potential [V]",
]
whole_cell_variables = ["Electrolyte concentration", "Electrolyte potential [V]"]


if export_data:
    dir_path = "results/2019_08_asymptotic_spme/data/figure_4"
    exporter = pybamm.ExportCSV(dir_path)

for C_rate in C_rates:

    param["Typical current [A]"] = (
        C_rate * 24 * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
    )

    # solve model
    solutions = [None] * len(models)
    t_eval = np.linspace(0, 0.17, 100)

    for i, model in enumerate(models):
        param.update_model(model, discs[i])
        solutions[i] = model.default_solver.solve(model, t_eval)

    # find time where discharge capacity is _ [A.h.m-2].
    dc_val = 8
    t, y = solutions[1].t, solutions[1].y

    discharge_cap = pybamm.ProcessedVariable(
        models[1].variables["Discharge capacity [A.h.m-2]"], t, y, discs[1].mesh
    )

    def bisection(t_left, t_right):
        t_mid = (t_left + t_right) / 2

        if np.abs(discharge_cap(t_mid) - dc_val) < 1e-6:
            return t_mid

        if (discharge_cap(t_mid) - dc_val) * (discharge_cap(t_left) - dc_val) < 0:
            return bisection(t_left, t_mid)
        else:
            return bisection(t_mid, t_right)

    t_out = bisection(0, t[-1])

    print(
        "For a ",
        C_rate,
        "C discharge, we have dc=",
        discharge_cap(t_out),
        " at time ",
        t_out,
    )

    if generate_plots:

        for i in [1, 0]:

            model = models[i]

            t, y = solutions[i].t, solutions[i].y

            c_s_n = pybamm.ProcessedVariable(
                model.variables["X-averaged negative particle concentration"],
                t,
                y,
                discs[i].mesh,
            )
            c_e = pybamm.ProcessedVariable(
                model.variables["Electrolyte concentration"], t, y, discs[i].mesh
            )
            c_s_p = pybamm.ProcessedVariable(
                model.variables["X-averaged positive particle concentration"],
                t,
                y,
                discs[i].mesh,
            )
            phi_s_n = pybamm.ProcessedVariable(
                model.variables["Negative electrode potential [V]"], t, y, discs[i].mesh
            )
            phi_e = pybamm.ProcessedVariable(
                model.variables["Electrolyte potential [V]"], t, y, discs[i].mesh
            )
            phi_s_p = pybamm.ProcessedVariable(
                model.variables["Positive electrode potential [V]"], t, y, discs[i].mesh
            )

            l_n = param.process_symbol(
                pybamm.standard_parameters_lithium_ion.l_n
            ).evaluate()
            l_p = param.process_symbol(
                pybamm.standard_parameters_lithium_ion.l_p
            ).evaluate()

            # x_n = np.linspace(0, l_n, 30)
            # x_p = np.linspace(1 - l_p, 1, 30)
            x = np.linspace(0, 1, 100)
            # r = np.linspace(0, 1, 30)

            x_n = discs[i].mesh["negative electrode"][0].nodes
            x_p = discs[i].mesh["positive electrode"][0].nodes
            r = discs[i].mesh["negative particle"][0].nodes

            if i == 0:
                c_s_n_xav = c_s_n(t_out, r=r)
                c_s_p_xav = c_s_p(t_out, r=r)
                linestyle = "--"
                label = None
                color = "k"
            elif i == 1:
                # still need to average the particle concentrations for dfn
                c_s_n(t_out, x=x_n, r=r)

                c_s_n_xav = np.mean(c_s_n(t_out, x=x_n, r=r), 0)
                c_s_p_xav = np.mean(c_s_p(t_out, x=x_p, r=r), 0)

                linestyle = "-"
                label = C_rate
                color = colour[C_rate]

            plt.subplot(2, 3, 1)
            plt.plot(r, c_s_n_xav, lw=2, linestyle=linestyle, color=color)

            plt.subplot(2, 3, 2)
            plt.plot(
                x, c_e(t_out, x=x), lw=2, label=label, linestyle=linestyle, color=color
            )
            plt.subplot(2, 3, 3)
            plt.plot(r, c_s_p_xav, lw=2, linestyle=linestyle, color=color)

            plt.subplot(2, 3, 4)
            plt.plot(x_n, phi_s_n(t_out, x=x_n), lw=2, linestyle=linestyle, color=color)

            plt.subplot(2, 3, 5)
            plt.plot(x, phi_e(t_out, x=x), lw=2, linestyle=linestyle, color=color)

            plt.subplot(2, 3, 6)
            plt.plot(x_p, phi_s_p(t_out, x=x_p), lw=2, linestyle=linestyle, color=color)

if generate_plots:

    plt.subplot(2, 3, 1)
    plt.xlabel("r", fontsize=15)
    plt.ylabel("c_s_n", fontsize=15)
    plt.axis((0, 1, 0, 1))

    plt.subplot(2, 3, 2)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("c_e", fontsize=15)
    plt.legend(fontsize=15)

    plt.subplot(2, 3, 3)
    plt.xlabel("r", fontsize=15)
    plt.ylabel("c_s_p", fontsize=15)
    plt.axis((0, 1, 0, 1))

    plt.subplot(2, 3, 4)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("phi_s_n", fontsize=15)

    plt.subplot(2, 3, 5)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("phi_e", fontsize=15)

    plt.subplot(2, 3, 6)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("phi_s_p", fontsize=15)

    plt.show()
