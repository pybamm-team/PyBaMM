import pybamm
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.close('all')
# set logging level
pybamm.set_logging_level("INFO")

# load (2+1D) SPM model
options = {"bc_options": {"dimensionality": 1}}
model = pybamm.lithium_ion.SPM(options)
model.check_well_posedness()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 10,
    var.x_s: 5,
    var.x_p: 10,
    var.r_n: 10,
    var.r_p: 10,
    var.y: 1,
    var.z: 5,
}
# depnding on number of points in y-z plane may need to increase recursion depth...
sys.setrecursionlimit(10000)
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model -- simulate one hour discharge
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_end = 3600 / tau.evaluate(0)
t_eval = np.linspace(0, t_end, 10)
solution = model.default_solver.solve(model, t_eval)

#e_conc = pybamm.ProcessedVariable(
#        model.variables['Electrolyte concentration [mol.m-3]'],
#        solution.t,
#        solution.y,
#        mesh=mesh,
#        )

# plot
#plot = pybamm.QuickPlot(model, mesh, solution)
#plot.dynamic_plot()

def plot_var(var, time=-1):
    variable = model.variables[var]
    len_x = len(mesh.combine_submeshes(*variable.domain))
    len_z = variable.shape[0] // len_x
    entries = np.empty((len_x, len_z, len(solution.t)))
    
    for idx in range(len(solution.t)):
        t = solution.t[idx]
        y = solution.y[:, idx]
        entries[:, :, idx] = np.reshape(
    		variable.evaluate(t, y), [len_x, len_z]
    	)
    plt.figure()
    for bat_id in range(len_x):
        plt.plot(range(len_z), entries[bat_id, :, time].flatten())
    plt.figure()
    plt.imshow(entries[:, :, time])

#plot_var(var="Electrolyte concentration")
plot_var(var="Interfacial current density", time=-1)
#plot_var(var="Current collector current density", time=[0])
#plot_var(var="Local current collector potential difference", time=[0])
