#
# Check convergence of pybamm model to "true" comsol solution (i.e. extremely fine mesh)
#

import pybamm
import os
import sys
import pickle
import shared
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

# set style
matplotlib.rc_file(
    "results/2019_xx_2plus1D_pouch/_matplotlibrc", use_default_template=True
)

# increase recursion limit for large expression trees
sys.setrecursionlimit(100000)
pybamm.set_logging_level("INFO")

# choose values to loop over and provide filenames
values = np.array([1e5, 1e6, 1e7, 1e8, 1e9]) / 4.758
filenames = [
    "input/comsol_results/comsol_1plus1D_sigma_1e5.pickle",
    "input/comsol_results/comsol_1plus1D_sigma_1e6.pickle",
    "input/comsol_results/comsol_1plus1D_sigma_1e7.pickle",
    "input/comsol_results/comsol_1plus1D_sigma_1e8.pickle",
    "input/comsol_results/comsol_1plus1D_sigma_1e9.pickle",
]

# load current collector and DFN models
cc_model = pybamm.current_collector.EffectiveResistance1D()
dfn_av = pybamm.lithium_ion.DFN({"thermal": "x-lumped"}, name="Average DFN")
dfn = pybamm.lithium_ion.DFN(
    {"current collector": "potential pair", "dimensionality": 1, "thermal": "x-lumped"},
    name="1+1D DFN",
)
models = {"Current collector": cc_model, "Average DFN": dfn_av, "1+1D DFN": dfn}

# parameters
param = dfn.default_parameter_values

# process model and geometry, and discretise
meshes = {}
discs = {}
for name, model in models.items():
    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)

    # set mesh
    var = pybamm.standard_spatial_vars
    submesh_types = model.default_submesh_types
    # set npts
    var = pybamm.standard_spatial_vars
    npts = 16
    var_pts = {
        var.x_n: npts,
        var.x_s: npts,
        var.x_p: npts,
        var.r_n: npts,
        var.r_p: npts,
        var.z: npts,
    }
    meshes[name] = pybamm.Mesh(geometry, submesh_types, var_pts)
    discs[name] = pybamm.Discretisation(meshes[name], model.default_spatial_methods)
    discs[name].process_model(model, check_model=False)


# solve models. Then compute "error"
errors = {
    "Negative current collector potential [V]": [None] * len(values),
    "Positive current collector potential [V]": [None] * len(values),
    "X-averaged negative particle surface concentration [mol.m-3]": [None]
    * len(values),
    "X-averaged positive particle surface concentration [mol.m-3]": [None]
    * len(values),
    "Current collector current density [A.m-2]": [None] * len(values),
    "X-averaged cell temperature [K]": [None] * len(values),
    "Terminal voltage [V]": [None] * len(values),
}
errors_bar = {
    "Negative current collector potential [V]": [None] * len(values),
    "Positive current collector potential [V]": [None] * len(values),
    "X-averaged negative particle surface concentration [mol.m-3]": [None]
    * len(values),
    "X-averaged positive particle surface concentration [mol.m-3]": [None]
    * len(values),
    "Current collector current density [A.m-2]": [None] * len(values),
    "X-averaged cell temperature [K]": [None] * len(values),
    "Terminal voltage [V]": [None] * len(values),
}
sigmas = [None] * len(values)

for i, val in enumerate(values):

    comsol_variables = pickle.load(open(filenames[i], "rb"))
    comsol_t = comsol_variables["time"]

    # update values
    param.update(
        {
            "Negative current collector conductivity [S.m-1]": val,
            "Positive current collector conductivity [S.m-1]": val,
        }
    )
    for name, model in models.items():
        param.update_model(model, discs[name])

    # solve
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
    time = comsol_t / tau
    solutions = {}
    for name, model in models.items():
        if name == "Current collector":
            solver = pybamm.AlgebraicSolver(tol=1e-6)
            solutions[name] = solver.solve(model)
        else:
            # solver
            solver = pybamm.CasadiSolver(
                atol=1e-6, rtol=1e-6, root_tol=1e-3, root_method="hybr", mode="fast"
            )
            solutions[name] = solver.solve(model, time)

    mesh = meshes["1+1D DFN"]
    cc_mesh = meshes["Current collector"]
    solution = solutions["1+1D DFN"]
    solution_1D = solutions["Average DFN"]
    cc_solution = solutions["Current collector"]

    # create comsol vars interpolated onto pybamm mesh to compare errors
    comsol_model = shared.make_comsol_model(comsol_variables, mesh, param, thermal=True)

    # compute "error" using times up to voltage cut off
    t = solutions["1+1D DFN"].t
    # Note: casadi doesnt support events so we find this time after the solve
    if isinstance(solver, pybamm.CasadiSolver):
        V_cutoff = param.evaluate(
            pybamm.standard_parameters_lithium_ion.voltage_low_cut_dimensional
        )
        voltage = pybamm.ProcessedVariable(
            models["1+1D DFN"].variables["Terminal voltage [V]"],
            solution.t,
            solution.y,
            mesh=mesh,
        )(time)
        # only use times up to the voltage cutoff
        voltage_OK = voltage[voltage > V_cutoff]
        t = t[0 : len(voltage_OK)]

    def compute_error(variable_name):
        domain = comsol_model.variables[variable_name].domain

        if domain == []:
            comsol_var = pybamm.ProcessedVariable(
                comsol_model.variables[variable_name], solution.t, solution.y, mesh=mesh
            )(t=t)
            pybamm_var = pybamm.ProcessedVariable(
                models["1+1D DFN"].variables[variable_name],
                solution.t,
                solution.y,
                mesh=mesh,
            )(t=t)
        else:
            z = mesh["current collector"][0].nodes
            comsol_var = pybamm.ProcessedVariable(
                comsol_model.variables[variable_name], solution.t, solution.y, mesh=mesh
            )(z=z, t=t)
            pybamm_var = pybamm.ProcessedVariable(
                models["1+1D DFN"].variables[variable_name],
                solution.t,
                solution.y,
                mesh=mesh,
            )(z=z, t=t)

        # Compute error in positive potential with respect to the voltage
        if variable_name == "Positive current collector potential [V]":
            comsol_var = comsol_var - pybamm.ProcessedVariable(
                comsol_model.variables["Terminal voltage [V]"],
                solution.t,
                solution.y,
                mesh=mesh,
            )(t=t)
            pybamm_var = pybamm_var - pybamm.ProcessedVariable(
                models["1+1D DFN"].variables["Terminal voltage [V]"],
                solution.t,
                solution.y,
                mesh=mesh,
            )(t=t)

        # compute RMS difference divided by RMS of comsol_var
        error = np.sqrt(np.nanmean((pybamm_var - comsol_var) ** 2)) / np.sqrt(
            np.nanmean((comsol_var) ** 2)
        )
        return error

    def compute_error_bar(variable_name):
        domain = comsol_model.variables[variable_name].domain

        if domain == []:
            comsol_var = pybamm.ProcessedVariable(
                comsol_model.variables[variable_name],
                solution.t,
                solution.y,
                mesh=cc_mesh,
            )(t=t)
        else:
            z = cc_mesh["current collector"][0].nodes
            comsol_var = pybamm.ProcessedVariable(
                comsol_model.variables[variable_name],
                solution.t,
                solution.y,
                mesh=cc_mesh,
            )(z=z, t=t)

        # Compute error in positive potential with respect to the voltage
        if variable_name == "Positive current collector potential [V]":
            comsol_var = comsol_var - pybamm.ProcessedVariable(
                comsol_model.variables["Terminal voltage [V]"],
                solution.t,
                solution.y,
                mesh=mesh,
            )(t=t)

        # compute pybamm vars for 1+1D bar model
        R_cc = param.process_symbol(
            cc_model.variables["Effective current collector resistance"]
        ).evaluate(t=cc_solution.t, y=cc_solution.y)[0][0]

        V_av_1D = pybamm.ProcessedVariable(
            models["Average DFN"].variables["Terminal voltage"],
            solution_1D.t,
            solution_1D.y,
            mesh=mesh,
        )
        I_av = pybamm.ProcessedVariable(
            models["Average DFN"].variables["Total current density"],
            solution_1D.t,
            solution_1D.y,
            mesh=mesh,
        )

        def V_av(t):
            return V_av_1D(t) - I_av(t) * R_cc

        pot_scale = param.evaluate(
            pybamm.standard_parameters_lithium_ion.potential_scale
        )
        U_ref = param.evaluate(
            pybamm.standard_parameters_lithium_ion.U_p_ref
        ) - param.evaluate(pybamm.standard_parameters_lithium_ion.U_n_ref)

        def V_av_dim(t):
            return U_ref + V_av(t) * pot_scale

        if variable_name == "Negative current collector potential [V]":
            potentials = cc_model.get_processed_potentials(
                cc_solution, cc_mesh, param, V_av, I_av
            )
            pybamm_var = potentials[variable_name](t, z)
        elif variable_name == "Positive current collector potential [V]":
            potentials = cc_model.get_processed_potentials(
                cc_solution, cc_mesh, param, V_av, I_av
            )
            pybamm_var = potentials[variable_name](t, z) - V_av_dim(t)
        elif variable_name == "Terminal voltage [V]":
            pybamm_var = V_av_dim(t)
        else:
            pybamm_var_1D = pybamm.ProcessedVariable(
                models["Average DFN"].variables[variable_name],
                solution_1D.t,
                solution_1D.y,
                mesh=mesh,
            )

            pybamm_var = np.transpose(
                np.repeat(pybamm_var_1D(t)[:, np.newaxis], len(z), axis=1)
            )

        # compute RMS difference divided by RMS of comsol_var
        error = np.sqrt(np.nanmean((pybamm_var - comsol_var) ** 2)) / np.sqrt(
            np.nanmean((comsol_var) ** 2)
        )

        return error

    # compute non-dim sigma (note sigma_cn=sigma_cp)
    sigmas[i] = param.evaluate(pybamm.standard_parameters_lithium_ion.sigma_cn)
    # compute errors
    for variable in errors.keys():
        try:
            errors[variable][i] = compute_error(variable)
        except KeyError:
            pass
        try:
            errors_bar[variable][i] = compute_error_bar(variable)
        except KeyError:
            pass


# set up figure
fig, ax = plt.subplots(1, 2, figsize=(6.4, 4))
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.8, top=0.93, wspace=0.33, hspace=0.5)
# fig.subplots_adjust(left=0.3, bottom=0.1, right=0.7, top=0.95, wspace=0.4, hspace=0.8)
labels = {
    "Negative current collector potential [V]": r"$\phi^*_{\mathrm{s,cn}}$",
    "Positive current collector potential [V]": r"$\phi^*_{\mathrm{s,cp}} - V^*$",
    "X-averaged negative particle surface concentration [mol.m-3]": r"$\bar{c}_{\mathrm{s,n,surf}}^*$",
    "X-averaged positive particle surface concentration [mol.m-3]": r"$\bar{c}_{\mathrm{s,p,surf}}^*$",
    "Current collector current density [A.m-2]": r"$\mathcal{I}^*$",
    "X-averaged cell temperature [K]": r"$\bar{T}^*$",
    "Terminal voltage [V]": r"$V^*$",
}
# loop of vals to plot
delta = param.evaluate(pybamm.standard_parameters_lithium_ion.delta)
sigmas = np.array(sigmas)

counter = 0
for variable in [
    "Negative current collector potential [V]",
    "Positive current collector potential [V]",
    "X-averaged negative particle surface concentration [mol.m-3]",
    "X-averaged positive particle surface concentration [mol.m-3]",
]:
    counter += 1
    # dummy points for colors to add to legend
    ax[1].plot(np.nan, np.nan, "o", color="C{}".format(counter), label=labels[variable])
    try:
        ax[0].plot(
            sigmas * delta ** 2,
            errors[variable],
            marker="o",
            linestyle="solid",
            markersize=7,
            fillstyle="none",
            color="C{}".format(counter),
        )
    except KeyError:
        pass
    try:
        ax[0].plot(
            sigmas * delta ** 2,
            errors_bar[variable],
            marker="x",
            linestyle="dotted",
            markersize=7,
            color="C{}".format(counter),
        )
    except KeyError:
        pass

for variable in [
    "Current collector current density [A.m-2]",
    "X-averaged cell temperature [K]",
    "Terminal voltage [V]",
]:
    counter += 1
    # dummy points for colors to add to legend
    ax[1].plot(np.nan, np.nan, "o", color="C{}".format(counter), label=labels[variable])
    try:
        ax[1].plot(
            sigmas * delta ** 2,
            errors[variable],
            marker="o",
            linestyle="solid",
            markersize=7,
            fillstyle="none",
            color="C{}".format(counter),
        )
    except KeyError:
        pass
    try:
        ax[1].plot(
            sigmas * delta ** 2,
            errors_bar[variable],
            marker="x",
            linestyle="dotted",
            markersize=7,
            color="C{}".format(counter),
        )
    except KeyError:
        pass

# labels and legend
ax[0].set_xlabel(r"$\sigma' = \delta^2 \sigma$")
ax[0].set_ylabel("RMS Error")
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[1].set_xlabel(r"$\sigma' = \delta^2 \sigma$")
ax[1].set_ylabel("RMS Error")
ax[1].set_xscale("log")
ax[1].set_yscale("log")

ax[0].set_xlim([1e-1, 1e4])
ax[0].set_ylim([1e-4, 1])
ax[0].set_xticks([1, 1e1, 1e2, 1e3, 1e4])
ax[0].set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e-2, 1e-1, 1])
ax[1].set_xlim([1e-1, 1e4])
ax[1].set_ylim([1e-6, 1])
ax[1].set_xticks([1, 1e1, 1e2, 1e3, 1e4])
ax[1].set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-2, 1e-1, 1])


leg1 = ax[1].legend(loc="lower left", bbox_to_anchor=(1.05, 0.1), borderaxespad=0.0)
# add dummy points for legend of styles
m_1plus1D, = ax[1].plot(np.nan, np.nan, "ko-", fillstyle="none")
m_DFNCC, = ax[1].plot(np.nan, np.nan, "kx:")
leg2 = ax[1].legend(
    [m_1plus1D, m_DFNCC],
    [r"$1+1$D", "DFNCC"],
    loc="lower left",
    bbox_to_anchor=(1.05, 0.8),
    borderaxespad=0.0,
)

ax[1].add_artist(leg1)

plt.savefig("RMSE_vs_sigma.pdf", format="pdf", dpi=1000)

plt.show()
