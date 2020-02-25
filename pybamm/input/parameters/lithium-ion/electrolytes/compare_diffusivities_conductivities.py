#
# Compare electrolyte diffusivities
#
import pybamm
import numpy as np
import matplotlib.pyplot as plt
from lipf6_Marquis2019.electrolyte_diffusivity_Capiglia1999 import (
    electrolyte_diffusivity_Capiglia1999,
)
from lipf6_Valoen2005.electrolyte_diffusivity_Valoen2005 import (
    electrolyte_diffusivity_Valoen2005,
)
from lipf6_Marquis2019.electrolyte_conductivity_Capiglia1999 import (
    electrolyte_conductivity_Capiglia1999,
)
from lipf6_Valoen2005.electrolyte_conductivity_Valoen2005 import (
    electrolyte_conductivity_Valoen2005,
)

param = pybamm.standard_parameters_lithium_ion
parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

c_e = pybamm.Vector(np.linspace(0, 3e3))
T = 300
c_e_eval = c_e.evaluate()


fig, axes = plt.subplots(1, 2)

# Diffusivities
D_e_Capiglia = parameter_values.process_symbol(
    electrolyte_diffusivity_Capiglia1999(c_e, T)
).evaluate()
D_e_Valoen = parameter_values.process_symbol(
    electrolyte_diffusivity_Valoen2005(c_e, T)
).evaluate()
axes[0].plot(c_e_eval, D_e_Capiglia, label="Capiglia")
axes[0].plot(c_e_eval, D_e_Valoen, label="Valoen")
axes[0].set_xlabel("Electrolyte concentration [mol/m$^3$]")
axes[0].set_ylabel("Electrolyte diffusivity [m$^2$/s]")
axes[0].legend()

# Conductivities
k_e_Capiglia = parameter_values.process_symbol(
    electrolyte_conductivity_Capiglia1999(c_e, T)
).evaluate()
k_e_Valoen = parameter_values.process_symbol(
    electrolyte_conductivity_Valoen2005(c_e, T)
).evaluate()
axes[1].plot(c_e_eval, k_e_Capiglia, label="Capiglia")
axes[1].plot(c_e_eval, k_e_Valoen, label="Valoen")
axes[1].set_xlabel("Electrolyte concentration [mol/m$^3$]")
axes[1].set_ylabel("Electrolyte conductivity [S/m]")
axes[1].legend()

fig.tight_layout()
plt.show()
