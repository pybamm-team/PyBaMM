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
L_cn = pybamm.Parameter("Negative current collector thickness [m]")
L_n = pybamm.Parameter("Negative electrode thickness [m]")
L_s = pybamm.Parameter("Separator thickness [m]")
L_p = pybamm.Parameter("Positive electrode thickness [m]")
L_cp = pybamm.Parameter("Positive current collector thickness [m]")
L_x = L_n + L_s + L_p  # Total distance between current collectors
L = L_cn + L_x + L_cp  # Total cell thickness
L_y = pybamm.Parameter("Electrode width [m]")
L_z = pybamm.Parameter("Electrode height [m]")
A_cc = L_y * L_z  # Area of current collector
A_cooling = pybamm.Parameter("Cell cooling surface area [m2]")
V_cell = pybamm.Parameter("Cell volume [m3]")

# Tab geometry
L_tab_n = pybamm.Parameter("Negative tab width [m]")
Centre_y_tab_n = pybamm.Parameter("Negative tab centre y-coordinate [m]")
Centre_z_tab_n = pybamm.Parameter("Negative tab centre z-coordinate [m]")
L_tab_p = pybamm.Parameter("Positive tab width [m]")
Centre_y_tab_p = pybamm.Parameter("Positive tab centre y-coordinate [m]")
Centre_z_tab_p = pybamm.Parameter("Positive tab centre z-coordinate [m]")
A_tab_n = L_tab_n * L_cn  # Area of negative tab
A_tab_p = L_tab_p * L_cp  # Area of negative tab


# Microscale geometry
a_n_dim = pybamm.Parameter("Negative electrode surface area to volume ratio [m-1]")
a_p_dim = pybamm.Parameter("Positive electrode surface area to volume ratio [m-1]")
R_n = pybamm.Parameter("Negative particle radius [m]")
R_p = pybamm.Parameter("Positive particle radius [m]")
b_e_n = pybamm.Parameter("Negative electrode Bruggeman coefficient (electrolyte)")
b_e_s = pybamm.Parameter("Separator Bruggeman coefficient (electrolyte)")
b_e_p = pybamm.Parameter("Positive electrode Bruggeman coefficient (electrolyte)")
b_s_n = pybamm.Parameter("Negative electrode Bruggeman coefficient (electrode)")
b_s_s = pybamm.Parameter("Separator Bruggeman coefficient (electrode)")
b_s_p = pybamm.Parameter("Positive electrode Bruggeman coefficient (electrode)")

# --------------------------------------------------------------------------------------
"Dimensionless Parameters"
# Macroscale Geometry
l_cn = L_cn / L_x
l_n = L_n / L_x
l_s = L_s / L_x
l_p = L_p / L_x
l_cp = L_cp / L_x
l_x = L_x / L_x
l_y = L_y / L_z
l_z = L_z / L_z
a_cc = l_y * l_z
a_cooling = A_cooling / (L_z ** 2)
v_cell = V_cell / (L_x * L_z ** 2)

l = L / L_x
delta = L_x / L_z  # Aspect ratio

# Tab geometry
l_tab_n = L_tab_n / L_z
centre_y_tab_n = Centre_y_tab_n / L_z
centre_z_tab_n = Centre_z_tab_n / L_z
l_tab_p = L_tab_p / L_z
centre_y_tab_p = Centre_y_tab_p / L_z
centre_z_tab_p = Centre_z_tab_p / L_z
