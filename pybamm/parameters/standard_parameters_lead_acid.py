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

"""
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
from pybamm import Parameter, Function
import pybamm.standard_parameters as sp

# --------------------------------------------------------------------------------------
"""Dimensional Parameters"""

# Electrolyte Properties
s_plus_n = Parameter("s_plus_n")
s_plus_p = Parameter("s_plus_p")

# Current collectors
H = Parameter("H")
W = Parameter("W")
A_cs = H * W  # Area of the current collectors [m2]
Ibar = Parameter("current scale")
ibar = Ibar / (8 * A_cs)  # Specified scale for the current [A.m-2]
Q = 17  # Capacity [Ah]
Crate = Ibar / Q  # C-rate [-]

# Microstructure
An = Parameter("An")  # Negative electrode surface area density [m-1] (or 1e4 or 1e6?)
Ap = Parameter("Ap")  # Positive electrode surface area density [m-1]
epsn_max = Parameter("epsn_max")  # Max porosity of negative electrode [-]
epss_max = Parameter("epss_max")  # Max porosity of separator [-]
epsp_max = Parameter("epsp_max")  # Max porosity of positive electrode [-]

# Stoichiometric coefficients
spn = Parameter("s_+")  # s_+ in the negative electrode [-]
spp = Parameter("s_+")  # s_+ in the positive electrode [-]

# Electrolyte physical properties
cmax = Parameter("cmax") * 1e3  # Maximum electrolye concentration [mol.m-3]
Vw = Parameter("Vw")  # Partial molar volume of water [m3.mol-1]
Vp = Parameter("Vp")  # Partial molar volume of cations [m3.mol-1]
Vn = Parameter("Vn")  # Partial molar volume of anions [m3.mol-1]
Ve = Vn + Vp  # Partial molar volume of electrolyte [m3.mol-1]
Mw = Parameter("Mw")  # Molar mass of water [kg.mol-1]
Mp = Parameter("Mp")  # Molar mass of cations [kg.mol-1]
Mn = Parameter("Mn")  # Molar mass of anions [kg.mol-1]
Me = Mn + Mp  # Molar mass of electrolyte [kg.mol-1]
DeltaVliqN = Vn - Vp  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]
DeltaVliqP = (
    2 * Vw - Vn - 3 * Vp
)  # Net Molar Volume consumed in electrolyte (neg) [m3.mol-1]

# Electrode physical properties
VPb = Parameter("VPb")  # Molar volume of lead [m3.mol-1]
VPbO2 = Parameter("VPbO2")  # Molar volume of lead dioxide [m3.mol-1]
VPbSO4 = Parameter("VPbSO4")  # Molar volume of lead sulfate [m3.mol-1]
DeltaVsurfN = VPb - VPbSO4  # Net Molar Volume consumed in neg electrode [m3.mol-1]
DeltaVsurfP = VPbSO4 - VPbO2  # Net Molar Volume consumed in pos electrode [m3.mol-1]
sigma_eff_n = sp.sigma_n * (1 - epsn_max) ** 1.5  # Effective lead conductivity [S/m-1]
sigma_eff_p = (
    sp.sigma_p * (1 - epsp_max) ** 1.5
)  # Effective lead dioxide conductivity [S/m-1]
d = Parameter("d")  # Pore size [m]

# Butler-Volmer
jref_n = Parameter("jref_n")  # Reference exchange-current density (neg) [A.m-2]
jref_p = Parameter("jref_p")  # Reference exchange-current density (pos) [A.m-2]
Cdl = Parameter("Cdl")  # Double-layer capacity [sp.F.m-2]
U_Pb_ref = Parameter("U_Pb_ref")  # Reference OCP in the lead [V]
U_PbO2_ref = Parameter("U_PbO2_ref")  # Reference OCP in the lead dioxide [V]

# Functions
D_dim = Function(None, "diffusivity_lead_acid")
rho_dim = Function(None, "density_lead_acid")
mu_dim = Function(None, "viscosity_lead_acid")

# --------------------------------------------------------------------------------------
"""Dimensionless Parameters"""

Cd = (
    (sp.L ** 2) / D_dim(cmax) / (cmax * sp.F * sp.L / ibar)
)  # Diffusional C-rate: diffusion timescale/discharge timescale
alpha = (sp.ne * Vw - Ve) * cmax  # Excluded volume fraction
sn = -(s_plus_n + sp.ne * sp.t_plus) / sp.ne  # Dimensionless rection rate (neg)
sp = -(s_plus_p + sp.ne * sp.t_plus) / sp.ne  # Dimensionless rection rate (pos)
iota_s_n = (
    sigma_eff_n * sp.R * sp.T / (sp.F * sp.L) / ibar
)  # Dimensionless lead conductivity
iota_s_p = (
    sigma_eff_p * sp.R * sp.T / (sp.F * sp.L) / ibar
)  # Dimensionless lead dioxide conductivity
iota_ref_n = jref_n / (
    ibar / (An * sp.L)
)  # Dimensionless exchange-current density (neg)
iota_ref_p = jref_p / (
    ibar / (Ap * sp.L)
)  # Dimensionless exchange-current density (pos)
beta_surf_n = -cmax * DeltaVsurfN / sp.ne  # Molar volume change (lead)
beta_surf_p = -cmax * DeltaVsurfP / sp.ne  # Molar volume change (lead dioxide)
beta_liq_n = -cmax * DeltaVliqN / sp.ne  # Molar volume change (electrolyte, neg)
beta_liq_p = -cmax * DeltaVliqP / sp.ne  # Molar volume change (electrolyte, pos)
beta_n = beta_surf_n + beta_liq_n  # Total molar volume change (neg)
beta_p = beta_surf_p + beta_liq_p  # Total molar volume change (pos)
omega_i = (
    cmax * Me / rho_dim(cmax) * (1 - Mw * Ve / Vw * Me)
)  # Diffusive kinematic relationship coefficient
omega_c = (
    cmax * Me / rho_dim(cmax) * (sp.t_plus + Mn / Me)
)  # Migrative kinematic relationship coefficient
gamma_dl_n = (
    Cdl * sp.R * sp.T * An * sp.L / (sp.F * ibar) / (cmax * sp.F * sp.L / ibar)
)  # Dimensionless double-layer capacity (neg)
gamma_dl_p = (
    Cdl * sp.R * sp.T * Ap * sp.L / (sp.F * ibar) / (cmax * sp.F * sp.L / ibar)
)  # Dimensionless double-layer capacity (pos)
voltage_cutoff = (
    sp.F / (sp.R * sp.T) * (sp.voltage_low_cut - (U_PbO2_ref - U_Pb_ref))
)  # Dimensionless voltage cut-off
U_rxn = ibar / (cmax * sp.F)  # Reaction velocity scale
pi_os = (
    mu_dim(cmax) * U_rxn * sp.L / (d ** 2 * sp.R * sp.T * cmax)
)  # Ratio of viscous pressure scale to osmotic pressure scale

# Initial conditions
q_init = Parameter("q_init")  # Initial SOC [-]
qmax = (
    (sp.Ln * epsn_max + sp.Ls * epss_max + sp.Lp * epsp_max) / sp.L / (sp - sn)
)  # Dimensionless max capacity
epsDeltan = beta_surf_n / sp.Ln * qmax
epsDeltap = beta_surf_p / sp.Lp * qmax
c_init = q_init
epsn_init = epsn_max - epsDeltan * (1 - q_init)  # Initial pororsity (neg) [-]
epss_init = epss_max  # Initial pororsity (sep) [-]
epsp_init = epsp_max - epsDeltap * (1 - q_init)  # Initial pororsity (pos) [-]
