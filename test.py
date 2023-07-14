solver_opt = 2
jacobian = 'sparse'  # sparse, dense, band, none
num_threads = 1

import pybamm
import numpy as np
import importlib

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

options = {'linear_solver': linear_solver, 'jacobian': jacobian, 'num_threads': num_threads}
klu_sol = pybamm.IDAKLUSolver(atol=1e-8, rtol=1e-8, options=options).solve(model, t_eval)
print(f"Solve time: {klu_sol.solve_time.value*1000} msecs")

# plot = pybamm.QuickPlot(klu_sol)
# plot.dynamic_plot()
