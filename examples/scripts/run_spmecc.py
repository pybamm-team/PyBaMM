import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")

var_pts = {
    pybamm.standard_spatial_vars.x_n: 5,
    pybamm.standard_spatial_vars.x_s: 5,
    pybamm.standard_spatial_vars.x_p: 5,
    pybamm.standard_spatial_vars.r_n: 5,
    pybamm.standard_spatial_vars.r_p: 5,
    pybamm.standard_spatial_vars.y: 5,
    pybamm.standard_spatial_vars.z: 5,
}

model = pybamm.lithium_ion.DFNCC()
extra_params = {"Typical timescale [s]": 3600}
solver = pybamm.IDAKLUSolver()
sim = pybamm.Simulation(
    model, solver=solver, update_parameter_values=extra_params, var_pts=var_pts
)
sim.solve()

variables = ["X-averaged negative particle surface concentration"]
built_vars = {var: sim.built_model.variables[var] for var in variables}
processed_vars = pybamm.post_process_variables(
    built_vars, sim.solution.t, sim.solution.y, mesh=sim.mesh
)

y = np.linspace(0, 1.5, 100)
z = np.linspace(0, 1, 100)

c_s_n_surf_xav = processed_vars["X-averaged negative particle surface concentration"](
    0, y=y, z=z
)

plt.pcolormesh(y, z, c_s_n_surf_xav, vmin=None, vmax=None, shading="gouraud")
plt.colorbar()
plt.show()

