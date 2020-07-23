import pybamm
import os
import numpy as np
import matplotlib.pyplot as plt
os.chdir(pybamm.__path__[0]+'/..')
model = pybamm.lithium_ion.SPM(build=False)
model.submodels["negative particle"] = pybamm.particle.FastSingleParticle(model.param, "Negative")
model.submodels["positive particle"] = pybamm.particle.FastSingleParticle(model.param, "Positive")
model.submodels["particle cracking"] = pybamm.particle_cracking.CrackPropagation(model.param, "Negative")
model.build_model()
param = model.default_parameter_values

import pandas as pd
mechanics = pd.read_csv("pybamm/input/parameters/lithium-ion/mechanicals/lico2_graphite_Ai2020/parameters.csv", 
                        index_col=0, comment="#", skip_blank_lines=True, header=None)[1][1:].dropna().astype(float).to_dict()
param.update(mechanics, check_already_exists=False)
# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 3600, 100)
solution = model.default_solver.solve(model, t_eval)

# extract voltage
stress_t_n_surf = solution['Negative particle surface tangential stress [Pa]']

# plot
plt.plot(solution["Time [h]"](solution.t), stress_t_n_surf(solution.t, x=0))
plt.xlabel(r'$t$')
plt.ylabel('Negative particle surface tangential stress')
plt.show()