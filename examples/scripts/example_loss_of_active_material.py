import pybamm
import os
import numpy as np
import matplotlib.pyplot as plt
os.chdir(pybamm.__path__[0]+'/..')
model = pybamm.lithium_ion.DFN(
    build=False, options={"particle": "Fickian diffusion", "sei":"solvent-diffusion limited", "sei film resistance":"distributed", "sei porosity change":False, "loss of active materials":True}
)
chemistry = pybamm.parameter_sets.Chen2020
param = pybamm.ParameterValues(chemistry=chemistry)

# add mechanical properties
import pandas as pd
mechanics = pd.read_csv("pybamm/input/parameters/lithium-ion/mechanicals/lico2_graphite_Ai2020/parameters.csv", 
                        index_col=0, comment="#", skip_blank_lines=True, header=None)[1][1:].dropna().astype(float).to_dict()
param.update(mechanics, check_already_exists=False)
# params.update({"Negative electrode Number of cracks per unit area of the particle [m-2]": 3.18e15/100})
param.update({"Negative electrode Cracking rate":3.9e-20*1000})
param.update({"Negative electrode LAM constant beta": 0})
param.update({"Positive electrode LAM constant beta": 1e-4})
total_cycles=2
experiment = pybamm.Experiment(
    ["Hold at 4.2 V until 1 mA",] +
    [
        "Discharge at 1C until 2.5 V",
        "Rest for 600 seconds",
        "Charge at 1C until 4.2 V",
        "Hold at 4.2 V until 1 mA",
    ] * total_cycles
)
sim1 = pybamm.Simulation(model, experiment=experiment,parameter_values=param)
solution = sim1.solve()
