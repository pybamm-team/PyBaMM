import pybamm
import numpy as np

# load parameters
param = pybamm.standard_parameters_lithium_ion

# load current collector model
cc_model = pybamm.current_collector.Ohm()

# create current collector mesh
cc_model.create_mesh(Ny=32, Nz=32, degree=1)

# assemble finite element matrices for the current collector model
cc_model.assemble()

# load choice of 1D model at each through-cell point
models = [None] * cc_model.N_dofs
for i in range(len(models)):
    models[i] = pybamm.lithium_ion.SPM()
    
# TO DO: get consisntent initial conditions

# manual timestepping, splitting between current collector and through-cell problems
t = 0.0  # initial time
t_final = (3600) / param.tau_d_star  # final time
dt = 15 / param.tau_d_star  # Coarse step size
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
        cc_model.update_current_values(current)
