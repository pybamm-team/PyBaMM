#
# Figure 4: DFN, SPMe internal states comparison
#
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import shared

generate_plots = True
export_data = True

spme = pybamm.lithium_ion.SPMe()
dfn = pybamm.lithium_ion.DFN()
models = shared.ModelGroup(spme, dfn)
models.process_parameters()

var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 20, var.x_p: 30, var.r_n: 15, var.r_p: 15}
var_pts = {var.x_n: 5, var.x_s: 5, var.x_p: 5, var.r_n: 5, var.r_p: 5}

models.discretise(var_pts)

x = np.linspace(0, 1, 100)
x_n = models.discretisations[0].mesh["negative electrode"][0].nodes
x_p = models.discretisations[0].mesh["positive electrode"][0].nodes
r = models.discretisations[0].mesh["negative particle"][0].nodes

C_rates = [0.1, 0.5, 1, 2, 3]
colour = {0.1: "r", 0.5: "b", 1: "g", 2: "m", 3: "y"}
t_eval = np.linspace(0, 0.17, 100)

negative_variables = [
    "Negative particle surface concentration",
    "Negative electrode potential [V]",
    "X-averaged negative particle concentration",
]
positive_variables = [
    "Positive particle surface concentration",
    "Positive electrode potential [V]",
    "X-averaged positive particle concentration",
]
whole_cell_variables = ["Electrolyte concentration", "Electrolyte potential [V]"]

# create exporters
dir_path = "results/2019_08_asymptotic_spme/data/figure_4"
c_s_n_exporter = pybamm.ExportCSV(dir_path)
c_e_exporter = pybamm.ExportCSV(dir_path)
c_s_p_exporter = pybamm.ExportCSV(dir_path)
phi_s_n_exporter = pybamm.ExportCSV(dir_path)
phi_e_exporter = pybamm.ExportCSV(dir_path)
phi_s_p_exporter = pybamm.ExportCSV(dir_path)

# add axis
c_s_n_exporter.add_array(r)
c_e_exporter.add_array(x)
c_s_p_exporter.add_array(r)
phi_s_n_exporter.add_array(x_n)
phi_e_exporter.add_array(x)
phi_s_p_exporter.add_array(x_p)

for C_rate in C_rates:
    update_parameters = {
        "Typical current [A]": C_rate
        * 24
        * models.parameters.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
    }
    models.solve(t_eval, update_parameters)

    # find time where discharge capacity is 8 [A.h.m-2].
    t_out = shared.find_time(8, models)

    neg_vars = models.process_variables(negative_variables)
    pos_vars = models.process_variables(positive_variables)
    whole_vars = models.process_variables(whole_cell_variables)

    plt.subplot(2, 3, 1)
    c_s_n = neg_vars[dfn]["X-averaged negative particle concentration"]
    c_s_n = neg_vars[dfn]["X-averaged negative particle concentration"]
    c_s_n_xav = np.mean(c_s_n(t_out, x=x_n, r=r), 0)

    plt.plot(r, c_s_n_xav, lw=2, color=colour[C_rate])

    plt.plot(
        r,
        neg_vars[spme]["X-averaged negative particle concentration"](t_out, r=r),
        lw=2,
        linestyle="--",
        color="k",
    )

    plt.subplot(2, 3, 2)
    plt.plot(
        x,
        whole_vars[dfn]["Electrolyte concentration"](t_out, x=x),
        lw=2,
        label="C" + str(C_rate),
        color=colour[C_rate],
    )
    plt.plot(
        x,
        whole_vars[spme]["Electrolyte concentration"](t_out, x=x),
        lw=2,
        linestyle="--",
        color="k",
    )

    plt.subplot(2, 3, 3)
    c_s_p = pos_vars[dfn]["X-averaged positive particle concentration"]
    c_s_p_xav = np.mean(c_s_p(t_out, x=x_p, r=r), 0)

    plt.plot(r, c_s_p_xav, lw=2, color=colour[C_rate])

    plt.plot(
        r,
        pos_vars[spme]["X-averaged positive particle concentration"](t_out, x=x_p, r=r),
        lw=2,
        linestyle="--",
        color="k",
    )

    plt.subplot(2, 3, 4)
    plt.plot(
        x_n,
        neg_vars[dfn]["Negative electrode potential [V]"](t_out, x=x_n),
        lw=2,
        color=colour[C_rate],
    )
    plt.plot(
        x_n,
        neg_vars[spme]["Negative electrode potential [V]"](t_out, x=x_n),
        lw=2,
        linestyle="--",
        color="k",
    )

    plt.subplot(2, 3, 5)
    plt.plot(
        x,
        whole_vars[dfn]["Electrolyte potential [V]"](t_out, x=x),
        lw=2,
        label="C" + str(C_rate),
        color=colour[C_rate],
    )
    plt.plot(
        x,
        whole_vars[spme]["Electrolyte potential [V]"](t_out, x=x),
        lw=2,
        linestyle="--",
        color="k",
    )

    plt.subplot(2, 3, 6)
    plt.plot(
        x_p,
        pos_vars[dfn]["Positive electrode potential [V]"](t_out, x=x_p),
        lw=2,
        color=colour[C_rate],
    )
    plt.plot(
        x_p,
        pos_vars[spme]["Positive electrode potential [V]"](t_out, x=x_p),
        lw=2,
        linestyle="--",
        color="k",
    )

    # add arrays to exporters
    c_s_n_exporter.add_array(c_s_n_xav)
    c_s_n_exporter.add_array(
        neg_vars[spme]["X-averaged negative particle concentration"](t_out, r=r)
    )
    c_e_exporter.add_array(whole_vars[dfn]["Electrolyte concentration"](t_out, x=x))
    c_e_exporter.add_array(whole_vars[spme]["Electrolyte concentration"](t_out, x=x))

    c_s_p_exporter.add_array(c_s_p_xav)
    c_s_p_exporter.add_array(
        pos_vars[spme]["X-averaged positive particle concentration"](t_out, r=r)
    )
    phi_s_n_exporter.add_array(
        neg_vars[dfn]["Negative electrode potential [V]"](t_out, x=x_n)
    )
    phi_s_n_exporter.add_array(
        neg_vars[spme]["Negative electrode potential [V]"](t_out, x=x_n)
    )
    phi_e_exporter.add_array(whole_vars[dfn]["Electrolyte potential [V]"](t_out, x=x))
    phi_e_exporter.add_array(whole_vars[spme]["Electrolyte potential [V]"](t_out, x=x))
    phi_s_p_exporter.add_array(
        pos_vars[dfn]["Positive electrode potential [V]"](t_out, x=x_p)
    )
    phi_s_n_exporter.add_array(
        pos_vars[spme]["Positive electrode potential [V]"](t_out, x=x_p)
    )

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

if generate_plots:
    plt.show()

if export_data:
    c_s_n_exporter.export("c_s_n")
    c_e_exporter.export("c_e")
    c_s_p_exporter.export("c_s_p")
    phi_s_n_exporter.export("phi_s_n")
    phi_e_exporter.export("phi_e")
    phi_s_p_exporter.export("phi_s_p")

