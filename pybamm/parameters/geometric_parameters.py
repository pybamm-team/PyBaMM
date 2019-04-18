#
# Geometric Parameters
#
"""
Standard geometric parameters
"""
import pybamm

# --------------------------------------------------------------------------------------
"Dimensional Parameters"
# Macroscale geometry
L_n = pybamm.Parameter("Negative electrode width")
L_s = pybamm.Parameter("Separator width")
L_p = pybamm.Parameter("Positive electrode width")
L_x = L_n + L_s + L_p  # Total cell width
L_y = pybamm.Parameter("Electrode depth")
L_z = pybamm.Parameter("Electrode height")
A_cc = L_y * L_z  # Area of current collector

# Microscale geometry
a_n_dim = pybamm.Parameter("Negative electrode surface area density")
a_p_dim = pybamm.Parameter("Positive electrode surface area density")
R_n = pybamm.Parameter("Negative particle radius")
R_p = pybamm.Parameter("Positive particle radius")
b = pybamm.Parameter("Bruggeman coefficient")


# --------------------------------------------------------------------------------------
"Dimensionless Parameters"
# Macroscale Geometry
l_n = L_n / L_x
l_s = L_s / L_x
l_p = L_p / L_x
l_y = L_y / L_z
l_z = L_z / L_z
