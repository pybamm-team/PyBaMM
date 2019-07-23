#
# Standard variables for the models
#
import pybamm

# Electrolyte concentration
c_e_n = pybamm.Variable(
    "Negative electrolyte concentration",
    domain="negative electrode",
    secondary_domain="current collector",
)
c_e_s = pybamm.Variable(
    "Separator electrolyte concentration",
    domain="separator",
    secondary_domain="current collector",
)
c_e_p = pybamm.Variable(
    "Positive electrolyte concentration",
    domain="positive electrode",
    secondary_domain="current collector",
)
c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

c_e_av = pybamm.Variable(
    "Average electrolyte concentration", domain="current collector"
)

# Electrolyte potential
phi_e_n = pybamm.Variable(
    "Negative electrolyte potential",
    domain="negative electrode",
    secondary_domain="current collector",
)
phi_e_s = pybamm.Variable(
    "Separator electrolyte potential",
    domain="separator",
    secondary_domain="current collector",
)
phi_e_p = pybamm.Variable(
    "Positive electrolyte potential",
    domain="positive electrode",
    secondary_domain="current collector",
)
phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

# Electrode potential
phi_s_n = pybamm.Variable(
    "Negative electrode potential",
    domain="negative electrode",
    secondary_domain="current collector",
)
phi_s_p = pybamm.Variable(
    "Positive electrode potential",
    domain="positive electrode",
    secondary_domain="current collector",
)

# Potential difference
delta_phi_n = pybamm.Variable(
    "Negative electrode surface potential difference",
    domain="negative electrode",
    secondary_domain="current collector",
)
delta_phi_p = pybamm.Variable(
    "Positive electrode surface potential difference",
    domain="positive electrode",
    secondary_domain="current collector",
)

delta_phi_n_av = pybamm.Variable(
    "Average negative electrode surface potential difference",
    domain="current collector",
)
delta_phi_p_av = pybamm.Variable(
    "Average positive electrode surface potential difference",
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


# Particle concentration
c_s_n = pybamm.SecondaryBroadcast(
    pybamm.Variable(
        "Negative particle concentration",
        "negative particle",
        secondary_domain="negative electrode",
    ),
    "current collector",
)
c_s_p = pybamm.SecondaryBroadcast(
    pybamm.Variable(
        "Positive particle concentration",
        "positive particle",
        secondary_domain="positive electrode",
    ),
    "current collector",
)
c_s_n_xav = pybamm.Variable(
    "X-average negative particle concentration", ["negative particle"]
)
c_s_p_xav = pybamm.Variable(
    "X-average positive particle concentration", ["positive particle"]
)

# Porosity
eps_n = pybamm.Variable(
    "Negative electrode porosity",
    domain="negative electrode",
    secondary_domain="current collector",
)
eps_s = pybamm.Variable(
    "Separator porosity", domain="separator", secondary_domain="current collector"
)
eps_p = pybamm.Variable(
    "Positive electrode porosity",
    domain="positive electrode",
    secondary_domain="current collector",
)
eps = pybamm.Concatenation(eps_n, eps_s, eps_p)

# Piecewise constant (for asymptotic models)
eps_n_pc = pybamm.Variable(
    "Average negative electrode porosity", domain="current collector"
)
eps_s_pc = pybamm.Variable("Average separator porosity", domain="current collector")
eps_p_pc = pybamm.Variable(
    "Average positive electrode porosity", domain="current collector"
)

eps_piecewise_constant = pybamm.Concatenation(
    pybamm.PrimaryBroadcast(eps_n_pc, "negative electrode"),
    pybamm.PrimaryBroadcast(eps_s_pc, "separator"),
    pybamm.PrimaryBroadcast(eps_p_pc, "positive electrode"),
)

# Electrolyte pressure
pressure_n = pybamm.Variable(
    "Negative electrolyte pressure",
    domain="negative electrode",
    secondary_domain="current collector",
)
pressure_s = pybamm.Variable(
    "Separator electrolyte pressure",
    domain="separator",
    secondary_domain="current collector",
)
pressure_p = pybamm.Variable(
    "Positive electrolyte pressure",
    domain="positive electrode",
    secondary_domain="current collector",
)
pressure = pybamm.Concatenation(pressure_n, pressure_s, pressure_p)

# Temperature
T_n = pybamm.Variable(
    "Negative electrode temperature",
    domain="negative electrode",
    secondary_domain="current collector",
)
T_s = pybamm.Variable(
    "Separator temperature", domain="separator", secondary_domain="current collector"
)
T_p = pybamm.Variable(
    "Positive electrode temperature",
    domain="positive electrode",
    secondary_domain="current collector",
)
T = pybamm.Concatenation(T_n, T_s, T_p)
T_av = pybamm.Variable("Average cell temperature")
