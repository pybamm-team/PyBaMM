#
# Standard variables for the models
#
import pybamm

# Electrolyte concentration
c_e_n = pybamm.Variable("Negative electrolyte concentration", ["negative electrode"])
c_e_s = pybamm.Variable("Separator electrolyte concentration", ["separator"])
c_e_p = pybamm.Variable("Positive electrolyte concentration", ["positive electrode"])
c_e = pybamm.Concatenation(c_e_n, c_e_s, c_e_p)

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

# Particle concentration
c_s_n = pybamm.Variable("Negative particle concentration", ["negative particle"])
c_s_p = pybamm.Variable("Positive particle concentration", ["positive particle"])

# Porosity
eps_n = pybamm.Variable("Negative electrode porosity", domain=["negative electrode"])
eps_s = pybamm.Variable("Separator porosity", domain=["separator"])
eps_p = pybamm.Variable("Positive electrode porosity", domain=["positive electrode"])
eps = pybamm.Concatenation(eps_n, eps_s, eps_p)
# Piecewise constant (for asymptotic models)
eps_n_pc = pybamm.Variable("Negative electrode porosity")
eps_s_pc = pybamm.Variable("Separator porosity")
eps_p_pc = pybamm.Variable("Positive electrode porosity")

eps_piecewise_constant = pybamm.Concatenation(
    pybamm.Broadcast(eps_n_pc, ["negative electrode"]),
    pybamm.Broadcast(eps_s_pc, ["separator"]),
    pybamm.Broadcast(eps_p_pc, ["positive electrode"]),
)
