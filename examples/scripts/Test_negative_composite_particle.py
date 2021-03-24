import pybamm
import os
import numpy as np
import matplotlib.pyplot as plt
import timeit
start = timeit.default_timer()
os.chdir(pybamm.__path__[0]+'/..')


model = pybamm.lithium_ion.DFN(
    options = {
        "particle": "negative composite", 
        "thermal": "lumped", 
    }
)
chemistry = pybamm.parameter_sets.Chen2020_composite
param = pybamm.ParameterValues(chemistry=chemistry)
param.update({"Upper voltage cut-off [V]": 4.21})
param.update({"Lower voltage cut-off [V]": 2.49})

exp = pybamm.Experiment(["Discharge at 1C until 2.5V"], use_simulation_setup_type='old')
sim1 = pybamm.Simulation(
    model, 
    experiment=exp,
    parameter_values=param,
    solver=pybamm.CasadiSolver(),
)
solution1 = sim1.solve()

stop = timeit.default_timer()
print('running time: ' + str(stop - start) +'s')