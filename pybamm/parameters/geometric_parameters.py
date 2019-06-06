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
L_n = pybamm.Parameter("Negative electrode width [m]")
L_s = pybamm.Parameter("Separator width [m]")
L_p = pybamm.Parameter("Positive electrode width [m]")
L_x = L_n + L_s + L_p  # Total cell width
L_y = pybamm.Parameter("Electrode depth [m]")
L_z = pybamm.Parameter("Electrode height [m]")
A_cc = L_y * L_z  # Area of current collector

# Microscale geometry
a_n_dim = pybamm.Parameter("Negative electrode surface area density [m-1]")
a_p_dim = pybamm.Parameter("Positive electrode surface area density [m-1]")
R_n = pybamm.Parameter("Negative particle radius [m]")
R_p = pybamm.Parameter("Positive particle radius [m]")
b = pybamm.Parameter("Bruggeman coefficient")


# --------------------------------------------------------------------------------------
"Dimensionless Parameters"
# Macroscale Geometry
l_n = L_n / L_x
l_s = L_s / L_x
l_p = L_p / L_x
l_y = L_y / L_z
l_z = L_z / L_z

delta = L_x / L_z
