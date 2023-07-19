import pybamm
import numpy as np
import matplotlib.pylab as plt
import importlib

solver_opt = 2
jacobian = 'sparse'  # sparse, dense, band, none
num_threads = 1
output_variables = [
    "Voltage [V]",
    "Time [min]",
    "Current [A]",
]
#output_variables = []
all_vars = False

input_parameters = {
    "Current function [A]": 0.15652,
    "Separator porosity": 0.47,
}

# check for loading errors
idaklu_spec = importlib.util.find_spec("pybamm.solvers.idaklu")
idaklu = importlib.util.module_from_spec(idaklu_spec)
idaklu_spec.loader.exec_module(idaklu)

# construct model
# pybamm.set_logging_level("INFO")
model = pybamm.lithium_ion.DFN()
# model.convert_to_format = 'jax'
geometry = model.default_geometry
param = model.default_parameter_values
param.update({key: "[input]" for key in input_parameters})
param.process_model(model)
param.process_geometry(geometry)
n = 100  # control the complexity of the geometry (increases number of solver states)
var_pts = {"x_n": n, "x_s": n, "x_p": n, "r_n": 10, "r_p": 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
t_eval = np.linspace(0, 3600, 100)

if solver_opt == 1:
    linear_solver = 'SUNLinSol_Dense'
if solver_opt == 2:
    linear_solver = 'SUNLinSol_KLU'
if solver_opt == 3:
    linear_solver = 'SUNLinSol_Band'
if solver_opt == 4:
    linear_solver = 'SUNLinSol_SPBCGS'
if solver_opt == 5:
    linear_solver = 'SUNLinSol_SPFGMR'
if solver_opt == 6:
    linear_solver = 'SUNLinSol_SPGMR'
if solver_opt == 7:
    linear_solver = 'SUNLinSol_SPTFQMR'
if solver_opt == 8:
    linear_solver = 'SUNLinSol_cuSolverSp_batchQR'
    jacobian = 'cuSparse_'

options = {
    'linear_solver': linear_solver,
    'jacobian': jacobian,
    'num_threads': num_threads,
}

if all_vars:
    output_variables = [m for m, (k, v) in
                        zip(model.variable_names(), model.variables.items())
                        if not isinstance(v, pybamm.ExplicitTimeIntegral)]
    left_out = [m for m, (k, v) in
                zip(model.variable_names(), model.variables.items())
                if isinstance(v, pybamm.ExplicitTimeIntegral)]
    print("ExplicitTimeIntegral variables:")
    print(left_out)

print("output_variables:")
print(output_variables)
print("\nInput parameters:")
print(input_parameters)

solver = pybamm.IDAKLUSolver(
    atol=1e-8, rtol=1e-8,
    options=options,
    output_variables=output_variables,
)

sol = solver.solve(
    model,
    t_eval,
    inputs=input_parameters,
    calculate_sensitivities=True,
)

print(f"Solve time: {sol.solve_time.value*1000} msecs")

if True:
    #output_variables = [
    #    "Voltage [V]",
    #    "Time [min]",
    #    "Current [A]",
    #]
    fig, axs = plt.subplots(len(output_variables), len(input_parameters)+1)
    for k, var in enumerate(output_variables):
        if False:
            axs[k,0].plot(t_eval, sol[var](t_eval))
            for paramk, param in enumerate(list(input_parameters.keys())):
                axs[k,paramk+1].plot(t_eval, sol[var].sensitivities[param]) # time, param, var
        else:
            axs[k,0].plot(t_eval, sol[var][:,0])
            for paramk, param in enumerate(list(input_parameters.keys())):
                axs[k,paramk+1].plot(t_eval, sol.yS[:,k,paramk]) # time, param, var
    plt.tight_layout()
    plt.show()
