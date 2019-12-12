import pybamm
import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

# increase recursion limit for large expression trees
sys.setrecursionlimit(100000)

pybamm.set_logging_level("INFO")

"-----------------------------------------------------------------------------"
"Set up figure"

fig, ax = plt.subplots(2, 1, figsize=(15, 8))
ax[0].set_xlabel("Discharge Capacity [Ah]")
ax[0].set_ylabel("Terminal voltage [V]")
ax[1].set_xlabel("Discharge Capacity [Ah]")
ax[1].set_ylabel("Error [mV]")


"-----------------------------------------------------------------------------"
"Build PyBamm Model"
C_rates = {"05": 0.5, "1": 1, "2": 2, "3": 3}

# load model and geometry
pybamm_model = pybamm.lithium_ion.DFN()
geometry = pybamm_model.default_geometry

# load parameters and process model and geometry
param = pybamm_model.default_parameter_values
param.process_model(pybamm_model)
param.process_geometry(geometry)

# create mesh
var = pybamm.standard_spatial_vars
# var_pts = {var.x_n: 101, var.x_s: 101, var.x_p: 101, var.r_n: 101, var.r_p: 101}
# var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20, var.r_n: 30, var.r_p: 30}
var_pts = {
    var.x_n: int(param.evaluate(pybamm.geometric_parameters.L_n / 1e-6)),
    var.x_s: int(param.evaluate(pybamm.geometric_parameters.L_s / 1e-6)),
    var.x_p: int(param.evaluate(pybamm.geometric_parameters.L_n / 1e-6)),
    var.r_n: int(param.evaluate(pybamm.geometric_parameters.R_n / 1e-7)),
    var.r_p: int(param.evaluate(pybamm.geometric_parameters.R_p / 1e-7)),
}
mesh = pybamm.Mesh(geometry, pybamm_model.default_submesh_types, var_pts)

# discretise model
spatial_methods = pybamm_model.default_spatial_methods
disc = pybamm.Discretisation(mesh, pybamm_model.default_spatial_methods)
disc.process_model(pybamm_model, check_model=False)

# solver
# solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6)
solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6, mode="fast")

"-----------------------------------------------------------------------------"
"Solve at different C_rates and plot against COMSOL solution"

counter = 0
interp_kind = "cubic"

for key, value in C_rates.items():
    # load comsol_model
    comsol_variables = pickle.load(
        open("input/comsol_results/comsol_isothermal_{}C.pickle".format(key), "rb")
    )
    comsol_t = comsol_variables["time"]
    comsol_voltage = interp.interp1d(
        comsol_t, comsol_variables["voltage"], kind=interp_kind
    )

    # update C_rate
    param.update({"C-rate": value})
    param.update_model(pybamm_model, disc)

    # solve
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
    time = comsol_t / tau  # use comsol time
    solution = solver.solve(pybamm_model, time)
    time = solution.t

    # plot using times up to voltage cut off
    # Note: casadi doesnt support events so we find this time after the solve
    if isinstance(solver, pybamm.CasadiSolver):
        V_cutoff = param.evaluate(
            pybamm.standard_parameters_lithium_ion.voltage_low_cut_dimensional
        )
        voltage = pybamm.ProcessedVariable(
            pybamm_model.variables["Terminal voltage [V]"],
            solution.t,
            solution.y,
            mesh=mesh,
        )(time)
        # only use times up to the voltage cutoff
        voltage_OK = voltage[voltage > V_cutoff]
        time = time[0 : len(voltage_OK)]

    # post-process pybamm solution
    pybamm_voltage = pybamm.ProcessedVariable(
        pybamm_model.variables["Terminal voltage [V]"], solution.t, solution.y, mesh
    )(t=time)

    # compute discharge_capacity
    dis_cap = pybamm.ProcessedVariable(
        pybamm_model.variables["Discharge capacity [A.h]"], solution.t, solution.y, mesh
    )(t=time)

    # add to plot
    ax[0].plot(
        dis_cap[0::10],
        comsol_voltage(time[0::10] * tau),
        "o",
        fillstyle="none",
        color="C{}".format(counter),
        label="COMSOL" if counter == 0 else "",
    )
    ax[0].plot(
        dis_cap,
        pybamm_voltage,
        "-",
        color="C{}".format(counter),
        label="PyBaMM ({}C)".format(value),
    )
    ax[1].plot(
        dis_cap,
        np.abs(pybamm_voltage - comsol_voltage(time * tau)) * 1000,
        "-",
        color="C{}".format(counter),
        label="{}C".format(value),
    )

    # increase counter for labelling
    counter += 1

"-----------------------------------------------------------------------------"
"Add legend and show plot"

ax[0].legend(loc="lower left")
ax[1].legend(loc="upper left")
plt.tight_layout()
plt.savefig("1D_voltage_C_rate.eps", format="eps", dpi=1000)
plt.show()
