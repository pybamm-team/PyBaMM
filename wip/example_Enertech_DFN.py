import pybamm
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(pybamm.__path__[0] + "/..")
# model = pybamm.lithium_ion.DFN(build=False,options = {"particle": "Fickian diffusion", "thermal": "lumped"})
model = pybamm.lithium_ion.DFN(
    build=False, options={"particle": "Fickian diffusion", "thermal": "lumped"}
)
param = model.default_parameter_values
chemistry = pybamm.parameter_sets.Ai2020
# param = pybamm.ParameterValues(chemistry=chemistry)
experiment = pybamm.Experiment(["Discharge at 1C until 3 V"])

# var = pybamm.standard_spatial_vars
# var_pts = {
#   var.x_n: 50,
#   var.x_s: 50,
#   var.x_p: 50,
#   var.r_n: 50,
#   var.r_p: 50,
# }
# import pandas as pd
# mechanics = pd.read_csv("pybamm/input/parameters/lithium-ion/mechanicals/lico2_graphite_Ai2020/parameters.csv", 
#                        index_col=0, comment="#", skip_blank_lines=True, header=None)[1][1:].dropna().astype(float).to_dict()
# param.update(mechanics, check_already_exists=False)

sim1 = pybamm.Simulation(model, experiment=experiment,parameter_values=param)
solution = sim1.solve()
print(param["Cell emissivity"])