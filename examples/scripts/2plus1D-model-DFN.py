#
# Script to solve "2+1D" battery models using fenics and pybamm DFN model
#
import pybamm

import numpy as np
import fenics
import matplotlib.pyplot as plt

# load parameters (should set up current collector submodel so that it can
# be processed, but lazy workaround for now)
set_of_parameters = pybamm.standard_parameters_lithium_ion  # parameter defintions
model = pybamm.lithium_ion.DFN()
param = model.default_parameter_values  # parameter values (from csv)
param.update({"Local potential difference": None})  # add parameter for local V

# load current collector model
cc_model = pybamm.current_collector.Ohm(set_of_parameters, param)

# create current collector mesh
cc_model.create_mesh(Ny=16, Nz=16, ny=2, nz=2, degree=1)

# assemble finite element matrices for the current collector model
cc_model.assemble()


# load and process choice of 1D model at each through-cell point
models = [None] * cc_model.n_dofs
for i in range(len(models)):
    models[i] = pybamm.lithium_ion.DFN()
    # Change to potentiostatic with paramater for local potential difference
    V_local = pybamm.Parameter("Local potential difference")
    model.boundary_conditions[model.variables["Positive electrode potential"]][
        "right"
    ] = (V_local, "Dirichlet")
for model in models:
    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# get initial voltage by assuming uniform through-cell current density
# (may need to then iterate on this at t=0)
current = param.process_symbol(
    set_of_parameters.I_typ / set_of_parameters.l_y / set_of_parameters.l_z
).evaluate(0, 0) * np.ones(cc_model.n_dofs)
cc_model.update_current(current)
cc_model.solve()
# plot
V_plot = fenics.plot(cc_model.voltage)
plt.colorbar(V_plot)
plt.xlabel(r"$y$", fontsize=22)
plt.ylabel(r"$z$", fontsize=22)
plt.title(r"$\mathcal{V}$", fontsize=24)
plt.show()

# manual timestepping, splitting between current collector and through-cell problems
t = 0.0  # initial time
t_final = 1  # final time
dt = 0.01  # coarse step size - need to invetsigate what size this should be
tol = 1e-3
max_iter = 5
solver = model.default_solver

while t < t_final:
    # increase time
    t += dt
    print(t)

    # reset iteration counter and voltage difference
    iter = 0
    cc_model.voltage_difference = 1

    while cc_model.voltage_difference > tol:

        # count iteration
        iter += 1
        print(iter)
        print(cc_model.voltage_difference)
        if iter > max_iter:
            raise pybamm.SolverError("maximum number of iterations exceeded")

        # update voltage
        cc_model.solve()
        V = cc_model.get_voltage(coarse=True)

        # compute new through-cell current
        # TO DO: loop over 1D models to solve here, changing BCs etc. appropriately
        # current is through-cell current from solution of 1D models
        for i in range(len(models)):
            # update local potential difference in boundary condition
            param["Local potential difference"] = V[i]
            param.update_model(models[i], disc)
            # solve
            t_eval = [t - dt, t]
            solver.solve(models[i], t_eval)
            print("solved model {} of {}".format(i + 1, len(models)))
            # get current
            i_e = pybamm.ProcessedVariable(
                model.variables["Electrolyte current density"],
                solver.t,
                solver.y,
                mesh=mesh,
            )
            current[i] = i_e(t, 0.5)

        # pass updated current to current collector model
        cc_model.update_current(current)

    # plot
    V_plot = fenics.plot(cc_model.voltage)
    plt.colorbar(V_plot)
    plt.xlabel(r"$y$", fontsize=22)
    plt.ylabel(r"$z$", fontsize=22)
    plt.title(r"$\mathcal{V}$", fontsize=24)
    plt.show()
