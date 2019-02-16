#
# Standard parameters for lead-acid battery models
#
"""
Standard Parameters for lead-acid battery models, to complement the ones given in
:module:`pybamm.standard_parameters`

Electrolyte Properties
----------------------
ce_typ
    Typical lithium ion concentration in electrolyte
De_typ
    Typical lithium ion diffusivity in the electrolyte
nu_plus
    Stoichiometry of hydrogen anions
nu_minus
    Stoichiometry of hydrogen sulfate anions
nu
    Stoichiometry of sulfuric acid
"""
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm

# --------------------------------------------------------------------------------------
"""Dimensional Parameters"""

# Current collectors / Electrical properties
H = pybamm.Parameter("H")
W = pybamm.Parameter("W")
A_cs = H * W  # Area of the current collectors [m2]
Ibar = pybamm.Parameter("current scale")
ibar = Ibar / (8 * A_cs)  # Specified scale for the current [A.m-2]
Q = 17  # Capacity [Ah]
Crate = Ibar / Q  # C-rate [-]

# Microstructure
An = pybamm.Parameter(
    "Anmax"
)  # Negative electrode surface area density [m-1] (or 1e4 or 1e6?)
Ap = pybamm.Parameter("Apmax")  # Positive electrode surface area density [m-1]
epsn_max = pybamm.Parameter("epsn_max")  # Max porosity of negative electrode [-]
epss_max = pybamm.Parameter("epss_max")  # Max porosity of separator [-]
epsp_max = pybamm.Parameter("epsp_max")  # Max porosity of positive electrode [-]

# Reactions
s_plus_n = pybamm.Parameter("s_plus_n")
s_plus_p = pybamm.Parameter("s_plus_p")
ne_n = pybamm.Parameter("ne_n")
ne_p = pybamm.Parameter("ne_p")

# Electrolyte physical properties
cmax = pybamm.Parameter("cmax")  # Maximum electrolye concentration [mol.m-3]
Vw = pybamm.Parameter("Vw")  # Partial molar volume of water [m3.mol-1]
V_plus = pybamm.Parameter("V_plus")  # Partial molar volume of cations [m3.mol-1]
V_minus = pybamm.Parameter("V_minus")  # Partial molar volume of anions [m3.mol-1]
Ve = V_minus + V_plus  # Partial molar volume of electrolyte [m3.mol-1]
Mw = pybamm.Parameter("Mw")  # Molar mass of water [kg.mol-1]
Mp = pybamm.Parameter("Mp")  # Molar mass of cations [kg.mol-1]
Mn = pybamm.Parameter("Mn")  # Molar mass of anions [kg.mol-1]
Me = Mn + Mp  # Molar mass of electrolyte [kg.mol-1]
DeltaVliq_n = (
    V_minus - V_plus
)  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]
DeltaVliq_p = (
    2 * Vw - V_minus - 3 * V_plus
)  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]
nu_plus = pybamm.Parameter("nu_plus")
nu_minus = pybamm.Parameter("nu_minus")
nu = nu_plus + nu_minus

# Electrode physical properties
VPb = pybamm.Parameter("VPb")  # Molar volume of lead [m3.mol-1]
VPbO2 = pybamm.Parameter("VPbO2")  # Molar volume of lead dioxide [m3.mol-1]
VPbSO4 = pybamm.Parameter("VPbSO4")  # Molar volume of lead sulfate [m3.mol-1]
DeltaVsurf_n = VPb - VPbSO4  # Net Molar Volume consumed in neg electrode [m3.mol-1]
DeltaVsurf_p = VPbSO4 - VPbO2  # Net Molar Volume consumed in pos electrode [m3.mol-1]
sigma_eff_n = (
    pybamm.standard_parameters.sigma_n * (1 - epsn_max) ** 1.5
)  # Effective lead conductivity [S/m-1]
sigma_eff_p = (
    pybamm.standard_parameters.sigma_p * (1 - epsp_max) ** 1.5
)  # Effective lead dioxide conductivity [S/m-1]
d = pybamm.Parameter("d")  # Pore size [m]

# Butler-Volmer
jref_n = pybamm.Parameter("jref_n")  # Reference exchange-current density (neg) [A.m-2]
jref_p = pybamm.Parameter("jref_p")  # Reference exchange-current density (pos) [A.m-2]
Cdl = pybamm.Parameter("Cdl")  # Double-layer capacity [F.m-2]
U_Pb_ref = pybamm.Parameter("U_Pb_ref")  # Reference OCP in the lead [V]
U_PbO2_ref = pybamm.Parameter("U_PbO2_ref")  # Reference OCP in the lead dioxide [V]

# --------------------------------------------------------------------------------------
"""Scales"""
Lx = pybamm.standard_parameters.Lx
F = pybamm.standard_parameters.F
R = pybamm.standard_parameters.R
T = pybamm.standard_parameters.T

concentration_scale = cmax
length_scale = Lx
time_scale = cmax * F * Lx / ibar  # discharge time scale
potential_scale = R * T / F
current_scale = ibar
interfacial_current_scale_neg = ibar / (An * Lx)
interfacial_current_scale_pos = ibar / (Ap * Lx)
velocity_scale = ibar / (cmax * F)  # Reaction velocity scale

# --------------------------------------------------------------------------------------
"""Functions"""


def dimensional_current(t):
    """Returns the dimensional current as a function of time """
    t = pybamm.t
    return ibar * pybamm.FunctionParameter("current function", t)


def dimensionless_current(t):
    """Returns the dimensionless current as a function of time """
    return pybamm.FunctionParameter("current function", t)


rho_dim = pybamm.Parameter("epsn_max")
mu_dim = pybamm.Parameter("epsn_max")


def D_dim(c):
    "Dimensional electrolyte diffusivity"
    return 1  # pybamm.FunctionParameter("electrolyte_diffusivity", c)


def D(c):
    "Dimensionless electrolyte diffusivity"
    return D_dim(c * concentration_scale) / D_dim(concentration_scale)


def U_Pb(c):
    "Dimensional open-circuit potential for lead electrode"
    return U_Pb_ref


def U_PbO2(c):
    "Dimensionless open-circuit potential for lead-dioxide electrode"
    return U_PbO2_ref


# --------------------------------------------------------------------------------------
"""Dimensionless Parameters"""

Cd = (
    (Lx ** 2) / D_dim(cmax) / time_scale
)  # Diffusional C-rate: diffusion timescale/discharge timescale
alpha = (nu * Vw - Ve) * cmax  # Excluded volume fraction
sn = (
    -(s_plus_n + ne_n * pybamm.standard_parameters.t_plus) / ne_n
)  # Dimensionless rection rate (neg)
sp = (
    -(s_plus_p + ne_p * pybamm.standard_parameters.t_plus) / ne_p
)  # Dimensionless rection rate (pos)
iota_s_n = (
    sigma_eff_n * potential_scale / current_scale
)  # Dimensionless lead conductivity
iota_s_p = (
    sigma_eff_p * potential_scale / current_scale
)  # Dimensionless lead dioxide conductivity
iota_ref_n = (
    jref_n / interfacial_current_scale_neg
)  # Dimensionless exchange-current density (neg)
iota_ref_p = (
    jref_p / interfacial_current_scale_pos
)  # Dimensionless exchange-current density (pos)
beta_surf_n = -cmax * DeltaVsurf_n / ne_n  # Molar volume change (lead)
beta_surf_p = -cmax * DeltaVsurf_p / ne_p  # Molar volume change (lead dioxide)
beta_liq_n = -cmax * DeltaVliq_n / ne_n  # Molar volume change (electrolyte, neg)
beta_liq_p = -cmax * DeltaVliq_p / ne_p  # Molar volume change (electrolyte, pos)
beta_n = beta_surf_n + beta_liq_n  # Total molar volume change (neg)
beta_p = beta_surf_p + beta_liq_p  # Total molar volume change (pos)
omega_i = (
    cmax * Me / rho_dim * (1 - Mw * Ve / Vw * Me)
)  # Diffusive kinematic relationship coefficient
omega_c = (
    cmax * Me / rho_dim * (pybamm.standard_parameters.t_plus + Mn / Me)
)  # Migrative kinematic relationship coefficient
gamma_dl_n = (
    Cdl * potential_scale / interfacial_current_scale_neg / time_scale
)  # Dimensionless double-layer capacity (neg)
gamma_dl_p = (
    Cdl * potential_scale / interfacial_current_scale_pos / time_scale
)  # Dimensionless double-layer capacity (pos)
voltage_cutoff = (
    1
    / potential_scale
    * (pybamm.standard_parameters.voltage_low_cut - (U_PbO2_ref - U_Pb_ref))
)  # Dimensionless voltage cut-off
pi_os = (
    mu_dim * velocity_scale * Lx / (d ** 2 * R * T * cmax)
)  # Ratio of viscous pressure scale to osmotic pressure scale


# Initial conditions
q_init = pybamm.Parameter("q_init")  # Initial SOC [-]
qmax = (
    (
        pybamm.standard_parameters.Ln * epsn_max
        + pybamm.standard_parameters.Ls * epss_max
        + pybamm.standard_parameters.Lp * epsp_max
    )
    / Lx
    / (sp - sn)
)  # Dimensionless max capacity
epsDelta_n = beta_surf_n / pybamm.standard_parameters.Ln * qmax
epsDelta_p = beta_surf_p / pybamm.standard_parameters.Lp * qmax
c_init = q_init
epsn_init = epsn_max - epsDelta_n * (1 - q_init)  # Initial pororsity (neg) [-]
epss_init = epss_max  # Initial pororsity (sep) [-]
epsp_init = epsp_max - epsDelta_p * (1 - q_init)  # Initial pororsity (pos) [-]


# Concatenated symbols
s = pybamm.PiecewiseConstant(sn, 0, sp)
beta_surf = pybamm.PiecewiseConstant(beta_surf_n, 0, beta_surf_p)
eps_init = pybamm.PiecewiseConstant(epsn_init, epss_init, epsp_init)
