import pybamm
import matplotlib.pyplot as plt
import numpy as np

param = pybamm.standard_parameters_lithium_ion
parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
sto = pybamm.Vector(np.linspace(0, 1, 200))
T = 300
sto_eval = sto.evaluate()
fig, ax = plt.subplots()
ocp_n = parameter_values.process_symbol(param.U_n_dimensional(sto, T)).evaluate()
ocp_p = parameter_values.process_symbol(param.U_p_dimensional(sto, T)).evaluate()
ax.plot(sto_eval, ocp_n, label="negative")
ax.plot(sto_eval, ocp_p, label="positive")
ax.set_xlabel("Stoichiometry")
ax.set_ylabel("U")
ax.legend()
plt.show()