#
# Standard parameters for SEI models
#

import pybamm

# --------------------------------------------------------------------------------------
# Dimensional parameters


def alpha(m_ratio, phi_s, phi_e, U_inner, U_outer):
    return pybamm.FunctionParameter(
        "Inner SEI reaction proportion", m_ratio, phi_s, phi_e, U_inner, U_outer
    )


V_bar_inner_dimensional = pybamm.Parameter("Inner SEI partial molar volume [m3.mol-1]")
V_bar_outer_dimensional = pybamm.Parameter("Outer SEI partial molar volume [m3.mol-1]")

m_sei_dimensional = pybamm.Parameter("SEI reaction exchange current density [A.m-2]")

R_sei_dimensional = pybamm.Parameter("SEI resistance per unit thickness [Ohm.m-1]")

D_sol_dimensional = pybamm.Parameter("Outer SEI solvent diffusivity [m2.s-1]")
c_sol_dimensional = pybamm.Parameter("Bulk solvent concentration [mol.m-3]")

m_ratio = pybamm.Parameter("Ratio of inner and outer SEI exchange current densities")

U_inner_dimensional = pybamm.Parameter("Inner SEI open-circuit potential [V]")
U_outer_dimensional = pybamm.Parameter("Outer SEI open-circuit potential [V]")

kappa_inner_dimensional = pybamm.Parameter("Inner SEI electron conductivity [S.m-1]")

D_li_dimensional = pybamm.Parameter(
    "Inner SEI lithium interstitial diffusivity [m2.s-1]"
)

c_li_0_dimensional = pybamm.Parameter(
    "Lithium interstitial reference concentration [mol.m-3]"
)

# --------------------------------------------------------------------------------------
# Dimensionless parameters

# write as C_SEI_reaction etc ...

