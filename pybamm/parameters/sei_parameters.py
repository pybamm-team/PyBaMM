#
# Standard parameters for SEI models
#

import pybamm

# --------------------------------------------------------------------------------------
# Dimensional parameters

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

L_inner_0_dim = pybamm.Parameter("Initial inner SEI thickness [m]")
L_outer_0_dim = pybamm.Parameter("Initial outer SEI thickness [m]")

L_sei_0_dim = L_inner_0_dim + L_outer_0_dim

# --------------------------------------------------------------------------------------
# Dimensionless parameters

U_n_ref = pybamm.standard_parameters_lithium_ion.U_n_ref
F = pybamm.standard_parameters_lithium_ion.F
R = pybamm.standard_parameters_lithium_ion.R
tau_discharge = pybamm.standard_parameters_lithium_ion.tau_discharge
T_ref = pybamm.standard_parameters_lithium_ion.T_ref

a_n = pybamm.standard_parameters_lithium_ion.a_n_dim
a_p = pybamm.standard_parameters_lithium_ion.a_p_dim
L_x = pybamm.standard_parameters_lithium_ion.L_x

i_typ = pybamm.electrical_parameters.i_typ

C_sei_reaction = (
    F * L_sei_0_dim / (m_sei_dimensional * tau_discharge * V_bar_inner_dimensional)
) * pybamm.exp(-(F * U_n_ref / (2 * R * T_ref)))

C_sei_solvent = L_sei_0_dim ** 2 / (
    c_sol_dimensional * V_bar_inner_dimensional * D_sol_dimensional * tau_discharge
)

C_sei_electron = (
    F ** 2
    * L_sei_0_dim ** 2
    / (kappa_inner_dimensional * V_bar_inner_dimensional * R * T_ref * tau_discharge)
)

C_sei_inter = L_sei_0_dim ** 2 / (
    D_li_dimensional * c_li_0_dimensional * V_bar_inner_dimensional * tau_discharge
)

U_inner_electron = F * U_inner_dimensional / R / T_ref

R_sei = F * i_typ * R_sei_dimensional * L_sei_0_dim / (a_n * L_x)

v_bar = V_bar_outer_dimensional / V_bar_inner_dimensional

L_inner_0 = L_inner_0_dim / L_sei_0_dim
L_outer_0 = L_outer_0_dim / L_sei_0_dim

# ratio of SEI reaction scale to intercalation reaction
Gamma_SEI_n = (F * L_sei_0_dim * a_n * L_x) / (
    V_bar_inner_dimensional * i_typ * tau_discharge
)

Gamma_SEI_p = (F * L_sei_0_dim * a_p * L_x) / (
    V_bar_inner_dimensional * i_typ * tau_discharge
)
