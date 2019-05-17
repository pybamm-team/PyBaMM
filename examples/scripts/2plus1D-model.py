#
# Script to solve "2+1D" battery models using fenics and pybamm
#
import pybamm

import numpy as np

# load parameters
param = pybamm.standard_parameters_lithium_ion

# load current collector model
cc_model = pybamm.current_collector.Ohm()

# create current collector mesh
cc_model.create_mesh(Ny=8, Nz=8, degree=1)

# assemble finite element matrices for the current collector model
cc_model.assemble()

# load and process choice of 1D model at each through-cell point
models = [None] * cc_model.N_dofs
for i in range(len(models)):
    models[i] = pybamm.lithium_ion.DFN()
for model in models:
    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

# get initial voltage by assuming uniform through-cell current density
# (may need to then iterate on this at t=0)
current = param.I_typ / param.l_y / param.l_z * np.ones(cc_model.N_dofs)
cc_model.update_current(current)
cc_model.solve()

# manual timestepping, splitting between current collector and through-cell problems
t = 0.0  # initial time
t_final = 1  # final time
dt = 0.01  # coarse step size - need to invetsigate what size this should be
tol = 1E-3

while t < t_final:

    # increase time
    t += dt

    while cc_model.voltage_difference > tol:

        # Update voltage
        cc_model.solve()

        # Compute new through-cell current
        # TO DO: loop over 1D models to solve here, changing BCs etc. appropriately
        # current is through-cell current from solution of 1D models
        # for i in range(len(models)):
        # current[i] = model.

        # cc_model.update_current(current)
