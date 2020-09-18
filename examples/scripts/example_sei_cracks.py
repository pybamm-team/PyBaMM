import pybamm
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(pybamm.__path__[0] + "/..")
# model = pybamm.lithium_ion.DFN(build=False,options = {"particle": "Fickian diffusion", "thermal": "lumped"})
model = pybamm.lithium_ion.DFN(
    build=False, options={"particle": "Fickian diffusion", "sei":"solvent-diffusion limited", "sei film resistance":"distributed", "sei porosity change":False}
)
model.submodels["negative particle cracking"] = pybamm.particle_cracking.CrackPropagation(
    model.param, "Negative"
)
model.submodels["negative sei on cracks"] = pybamm.sei.SEIonCracks(
    model.param, "Negative"
)
# model.build_model()
param = model.default_parameter_values
chemistry = pybamm.parameter_sets.Chen2020
param = pybamm.ParameterValues(chemistry=chemistry)
param.update({"Initial concentration in negative electrode [mol.m-3]": 29866})
param.update({"Initial concentration in positive electrode [mol.m-3]": 14778})
param.update({"Maximum concentration in negative electrode [mol.m-3]": 33133})
param.update({"Maximum concentration in positive electrode [mol.m-3]": 56840})
param.update({"Negative electrode diffusivity [m2.s-1]": 6e-15}) #6e-15 good value
param.update({"Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 0.1, 
"Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 0.1}, check_already_exists=False)
param.update({"Total heat transfer coefficient [W.m-2.K-1]": 0.1})
param.update({"Ambient temperature [K]": 298.15})
param.update({"Initial temperature [K]": 298.15})
param.update({"Initial inner SEI thickness [m]": 0})
param.update({"Inner SEI reaction proportion": 0})
param.update({"Outer SEI solvent diffusivity [m2.s-1]": 2.5e-22*1000})
param2 = param

# add mechanical properties
import pandas as pd
mechanics = pd.read_csv("pybamm/input/parameters/lithium-ion/mechanicals/lico2_graphite_Ai2020/parameters.csv", 
                        index_col=0, comment="#", skip_blank_lines=True, header=None)[1][1:].dropna().astype(float).to_dict()
param.update(mechanics, check_already_exists=False)
# params.update({"Negative electrode Number of cracks per unit area of the particle [m-2]": 3.18e15/100})
param.update({"Negative electrode Cracking rate":3.9e-20*1000})

total_cycles=3
experiment = pybamm.Experiment(
    [
        "Discharge at C/10 for 10 hours or until 3.3 V",
        #"Rest for 1 hour",
        #"Charge at 1 A until 4.1 V",
        #"Hold at 4.1 V until 50 mA",
        #"Rest for 1 hour",
    ] * total_cycles
)
sim1 = pybamm.Simulation(model, experiment=experiment,parameter_values=param)
#solution1 = sim1.solve()

# Use the default setup without sei-crack model
#model2 = pybamm.lithium_ion.DFN(
#     build=True, options={"particle": "Fickian diffusion", "sei":"solvent-diffusion limited", "sei film resistance":"distributed", "sei porosity change":False}
#)
#sim2 = pybamm.Simulation(model2, experiment=experiment,parameter_values=param2)
#solution2 = sim2.solve()

# plot results
# cycle_number1 = []
# Qdis_delta1 = []
# Qdis_delta2 = []
# for i in range(total_cycles):
#     t1 = solution1.sub_solutions[i*5]["Time [h]"].entries
#     Qdis1 = solution1.sub_solutions[i*5]["Discharge capacity [A.h]"].entries
#     Qdis_delta1.append(Qdis1[-1]-Qdis1[-0])
#     cycle_number1.append(i)
#     t2 = solution2.sub_solutions[i*5]["Time [h]"].entries
#     Qdis2 = solution2.sub_solutions[i*5]["Discharge capacity [A.h]"].entries
#     Qdis_delta2.append(Qdis2[-1]-Qdis2[-0])
# plt.plot(cycle_number1, Qdis_delta1,label='with sei-cracks')
# plt.plot(cycle_number1, Qdis_delta2,label='without sei-cracks')
# plt.xlabel("Cycle nunber")
# plt.ylabel("Discharge capacity [A.h]")
# plt.legend()
# plt.show()

