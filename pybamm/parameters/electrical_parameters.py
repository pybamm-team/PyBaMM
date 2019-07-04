#
# Standard electrical parameters
#
import pybamm


def abs_non_zero(x):
    if x == 0:  # pragma: no cover
        return 1
    else:
        return abs(x)


# --------------------------------------------------------------------------------------
# Dimensional Parameters
I_typ = pybamm.Parameter("Typical current [A]")
Q = pybamm.Parameter("Cell capacity [A.h]")
C_rate = abs(I_typ / Q)
n_electrodes_parallel = pybamm.Parameter(
    "Number of electrodes connected in parallel to make a cell"
)
i_typ = pybamm.Function(
    abs_non_zero, (I_typ / (n_electrodes_parallel * pybamm.geometric_parameters.A_cc))
)
voltage_low_cut_dimensional = pybamm.Parameter("Lower voltage cut-off [V]")
voltage_high_cut_dimensional = pybamm.Parameter("Upper voltage cut-off [V]")

# Current as a function of *dimensional* time. The below is overwritten in
# standard_parameters_lithium_ion.py and standard_parameters_lead_acid.py
# to use the correct timescale used for non-dimensionalisation. For a base model,
# the user may provide the typical timescale as a parameter.
timescale = pybamm.Parameter("Typical timescale [s]")
dimensional_current_with_time = pybamm.FunctionParameter(
    "Current function", pybamm.t * timescale
)
dimensional_current_density_with_time = i_typ * (dimensional_current_with_time / I_typ)
current_with_time = dimensional_current_with_time / I_typ
