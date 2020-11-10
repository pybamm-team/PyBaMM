import pybamm
import os
import numpy as np
import matplotlib.pyplot as plt
os.chdir(pybamm.__path__[0]+'/..')
model = pybamm.lithium_ion.DFN(
    build=False, options={"particle": "Fickian diffusion", "sei":"solvent-diffusion limited", "sei film resistance":"distributed", "sei porosity change":True}
)
chemistry = pybamm.parameter_sets.Chen2020
param = pybamm.ParameterValues(chemistry=chemistry)
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
t_all = solution["Time [s]"].entries
v_all = solution["Terminal voltage [V]"].entries
I_if_p = solution["Sum of x-averaged negative electrode interfacial current densities"].entries
I_if_n = solution["Sum of x-averaged positive electrode interfacial current densities"].entries
np.savetxt('test_LAM_data_orign.csv', (t_all, v_all, I_if_p, I_if_n), delimiter=',')