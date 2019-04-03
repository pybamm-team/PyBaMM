#
# Geometric Parameters
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

# --------------------------------------------------------------------------------------
"Dimensional Parameters"
# Electrical
I_typ = pybamm.Parameter("Typical current density")
Q = pybamm.Parameter("Cell capacity")
C_rate = I_typ / Q
n_electrodes_parallel = pybamm.Parameter(
    "Number of electrodes connected in parallel to make a cell"
)
i_typ = I_typ / (n_electrodes_parallel * pybamm.geometric_parameters.A_cc)
voltage_low_cut_dimensional = pybamm.Parameter("Lower voltage cut-off")
voltage_high_cut_dimensional = pybamm.Parameter("Upper voltage cut-off")
current_with_time = pybamm.FunctionParameter("Current function", pybamm.t)
dimensional_current_with_time = i_typ * current_with_time


# --------------------------------------------------------------------------------------
"Dimensionless Parameters"
