#
# Geometric Parameters
#
"""
Standard electrical parameters
"""
import pybamm


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def abs_non_zero(x):
    if x == 0:
        return 1
    else:
        return abs(x)


# --------------------------------------------------------------------------------------
"Dimensional Parameters"
# Electrical
I_typ = pybamm.Parameter("Typical current density")
Q = pybamm.Parameter("Cell capacity")
C_rate = abs(I_typ / Q)
n_electrodes_parallel = pybamm.Parameter(
    "Number of electrodes connected in parallel to make a cell"
)
i_typ = pybamm.Function(
    abs_non_zero, (I_typ / (n_electrodes_parallel * pybamm.geometric_parameters.A_cc))
)
voltage_low_cut_dimensional = pybamm.Parameter("Lower voltage cut-off")
voltage_high_cut_dimensional = pybamm.Parameter("Upper voltage cut-off")
current_with_time = pybamm.FunctionParameter(
    "Current function", pybamm.t
) * pybamm.Function(sign, I_typ)
dimensional_current_with_time = i_typ * current_with_time
