#
# Standard variables for the models
#
import pybamm

# Electrolyte concentration
c_e_n = pybamm.Variable("Negative electrolyte concentration", ["negative electrode"])
c_e_s = pybamm.Variable("Separator electrolyte concentration", ["separator"])
c_e_p = pybamm.Variable("Positive electrolyte concentration", ["positive electrode"])
c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

c_e_av = pybamm.Variable("Average electrolyte concentration")

# Electrolyte potential
phi_e_n = pybamm.Variable("Negative electrolyte potential", ["negative electrode"])
phi_e_s = pybamm.Variable("Separator electrolyte potential", ["separator"])
phi_e_p = pybamm.Variable("Positive electrolyte potential", ["positive electrode"])
phi_e = pybamm.Concatenation(phi_e_n, phi_e_s, phi_e_p)

# Electrode potential
phi_s_n = pybamm.Variable("Negative electrode potential", ["negative electrode"])
phi_s_p = pybamm.Variable("Positive electrode potential", ["positive electrode"])

# Potential difference
delta_phi_n = pybamm.Variable(
    "Negative electrode surface potential difference", ["negative electrode"]
)
delta_phi_p = pybamm.Variable(
    "Positive electrode surface potential difference", ["positive electrode"]
)

delta_phi_n_av = pybamm.Variable(
    "Average negative electrode surface potential difference"
)
delta_phi_p_av = pybamm.Variable(
    "Average positive electrode surface potential difference"
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
c_s_n = pybamm.Variable("Negative particle concentration", ["negative particle"])
c_s_p = pybamm.Variable("Positive particle concentration", ["positive particle"])
c_s_n_xav = pybamm.Variable(
    "X-average negative particle concentration", ["negative particle"]
)
c_s_p_xav = pybamm.Variable(
    "X-average positive particle concentration", ["positive particle"]
)

# Porosity
eps_n = pybamm.Variable("Negative electrode porosity", domain=["negative electrode"])
eps_s = pybamm.Variable("Separator porosity", domain=["separator"])
eps_p = pybamm.Variable("Positive electrode porosity", domain=["positive electrode"])
eps = pybamm.Concatenation(eps_n, eps_s, eps_p)

# Piecewise constant (for asymptotic models)
eps_n_pc = pybamm.Variable("Average negative electrode porosity")
eps_s_pc = pybamm.Variable("Average separator porosity")
eps_p_pc = pybamm.Variable("Average positive electrode porosity")

eps_piecewise_constant = pybamm.Concatenation(
    pybamm.Broadcast(eps_n_pc, ["negative electrode"]),
    pybamm.Broadcast(eps_s_pc, ["separator"]),
    pybamm.Broadcast(eps_p_pc, ["positive electrode"]),
)

# Electrolyte pressure
pressure_n = pybamm.Variable("Negative electrolyte pressure", ["negative electrode"])
pressure_s = pybamm.Variable("Separator electrolyte pressure", ["separator"])
pressure_p = pybamm.Variable("Positive electrolyte pressure", ["positive electrode"])
pressure = pybamm.Concatenation(pressure_n, pressure_s, pressure_p)

# Temperature
T_n = pybamm.Variable("Negative electrode temperature", ["negative electrode"])
T_s = pybamm.Variable("Separator temperature", ["separator"])
T_p = pybamm.Variable("Positive electrode temperature", ["positive electrode"])
T = pybamm.Concatenation(T_n, T_s, T_p)
T_av = pybamm.Variable("Average cell temperature")
