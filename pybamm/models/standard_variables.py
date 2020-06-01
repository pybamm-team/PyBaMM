#
# Standard variables for the models
#
import pybamm

# Electrolyte concentration
c_e_n = pybamm.Variable(
    "Negative electrolyte concentration",
    domain="negative electrode",
    auxiliary_domains={"secondary": "current collector"},
)
c_e_s = pybamm.Variable(
    "Separator electrolyte concentration",
    domain="separator",
    auxiliary_domains={"secondary": "current collector"},
)
c_e_p = pybamm.Variable(
    "Positive electrolyte concentration",
    domain="positive electrode",
    auxiliary_domains={"secondary": "current collector"},
)
c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

c_e_av = pybamm.Variable(
    "X-averaged electrolyte concentration", domain="current collector"
)

# Electrolyte potential
phi_e_n = pybamm.Variable(
    "Negative electrolyte potential",
    domain="negative electrode",
    auxiliary_domains={"secondary": "current collector"},
)
phi_e_s = pybamm.Variable(
    "Separator electrolyte potential",
    domain="separator",
    auxiliary_domains={"secondary": "current collector"},
)
phi_e_p = pybamm.Variable(
    "Positive electrolyte potential",
    domain="positive electrode",
    auxiliary_domains={"secondary": "current collector"},
)
phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

# Electrode potential
phi_s_n = pybamm.Variable(
    "Negative electrode potential",
    domain="negative electrode",
    auxiliary_domains={"secondary": "current collector"},
)
phi_s_p = pybamm.Variable(
    "Positive electrode potential",
    domain="positive electrode",
    auxiliary_domains={"secondary": "current collector"},
)

# Potential difference
delta_phi_n = pybamm.Variable(
    "Negative electrode surface potential difference",
    domain="negative electrode",
    auxiliary_domains={"secondary": "current collector"},
)
delta_phi_p = pybamm.Variable(
    "Positive electrode surface potential difference",
    domain="positive electrode",
    auxiliary_domains={"secondary": "current collector"},
)

delta_phi_n_av = pybamm.Variable(
    "X-averaged negative electrode surface potential difference",
    domain="current collector",
)
delta_phi_p_av = pybamm.Variable(
    "X-averaged positive electrode surface potential difference",
    domain="current collector",
)

# current collector variables
phi_s_cn = pybamm.Variable(
    "Negative current collector potential", domain="current collector"
)
phi_s_cp = pybamm.Variable(
    "Positive current collector potential", domain="current collector"
)
i_boundary_cc = pybamm.Variable(
    "Current collector current density", domain="current collector"
)
phi_s_cn_composite = pybamm.Variable(
    "Composite negative current collector potential", domain="current collector"
)
phi_s_cp_composite = pybamm.Variable(
    "Composite positive current collector potential", domain="current collector"
)
i_boundary_cc_composite = pybamm.Variable(
    "Composite current collector current density", domain="current collector"
)


# Particle concentration
c_s_n = pybamm.Variable(
    "Negative particle concentration",
    domain="negative particle",
    auxiliary_domains={
        "secondary": "negative electrode",
        "tertiary": "current collector",
    },
)
c_s_p = pybamm.Variable(
    "Positive particle concentration",
    domain="positive particle",
    auxiliary_domains={
        "secondary": "positive electrode",
        "tertiary": "current collector",
    },
)
c_s_n_xav = pybamm.Variable(
    "X-averaged negative particle concentration",
    domain="negative particle",
    auxiliary_domains={"secondary": "current collector"},
)
c_s_p_xav = pybamm.Variable(
    "X-averaged positive particle concentration",
    domain="positive particle",
    auxiliary_domains={"secondary": "current collector"},
)
c_s_n_surf = pybamm.Variable(
    "Negative particle surface concentration",
    domain="negative electrode",
    auxiliary_domains={"secondary": "current collector"},
)
c_s_p_surf = pybamm.Variable(
    "Positive particle surface concentration",
    domain="positive electrode",
    auxiliary_domains={"secondary": "current collector"},
)
c_s_n_surf_xav = pybamm.Variable(
    "X-averaged negative particle surface concentration", domain="current collector"
)
c_s_p_surf_xav = pybamm.Variable(
    "X-averaged positive particle surface concentration", domain="current collector"
)


# Porosity
eps_n = pybamm.Variable(
    "Negative electrode porosity",
    domain="negative electrode",
    auxiliary_domains={"secondary": "current collector"},
)
eps_s = pybamm.Variable(
    "Separator porosity",
    domain="separator",
    auxiliary_domains={"secondary": "current collector"},
)
eps_p = pybamm.Variable(
    "Positive electrode porosity",
    domain="positive electrode",
    auxiliary_domains={"secondary": "current collector"},
)
eps = pybamm.Concatenation(eps_n, eps_s, eps_p)

# Piecewise constant (for asymptotic models)
eps_n_pc = pybamm.Variable(
    "X-averaged negative electrode porosity", domain="current collector"
)
eps_s_pc = pybamm.Variable("X-averaged separator porosity", domain="current collector")
eps_p_pc = pybamm.Variable(
    "X-averaged positive electrode porosity", domain="current collector"
)

eps_piecewise_constant = pybamm.Concatenation(
    pybamm.PrimaryBroadcast(eps_n_pc, "negative electrode"),
    pybamm.PrimaryBroadcast(eps_s_pc, "separator"),
    pybamm.PrimaryBroadcast(eps_p_pc, "positive electrode"),
)

# Temperature
T_cn = pybamm.Variable(
    "Negative currents collector temperature", domain="current collector"
)
T_n = pybamm.Variable(
    "Negative electrode temperature",
    domain="negative electrode",
    auxiliary_domains={"secondary": "current collector"},
)
T_s = pybamm.Variable(
    "Separator temperature",
    domain="separator",
    auxiliary_domains={"secondary": "current collector"},
)
T_p = pybamm.Variable(
    "Positive electrode temperature",
    domain="positive electrode",
    auxiliary_domains={"secondary": "current collector"},
)
T_cp = pybamm.Variable(
    "Positive currents collector temperature", domain="current collector"
)
T = pybamm.Concatenation(T_n, T_s, T_p)
T_av = pybamm.Variable("X-averaged cell temperature", domain="current collector")
T_vol_av = pybamm.Variable("Volume-averaged cell temperature")


# SEI variables
L_inner_av = pybamm.Variable(
    "X-averaged inner SEI thickness", domain="current collector"
)
L_inner = pybamm.Variable(
    "Inner SEI thickness",
    domain=["negative electrode"],
    auxiliary_domains={"secondary": "current collector"},
)
L_outer_av = pybamm.Variable(
    "X-averaged outer SEI thickness", domain="current collector"
)
L_outer = pybamm.Variable(
    "Outer SEI thickness",
    domain=["negative electrode"],
    auxiliary_domains={"secondary": "current collector"},
)

