#
# Standard thermal parameters
#
import pybamm

# --------------------------------------------------------------------------------------
# Dimensional Parameters
I_typ = pybamm.Parameter("Typical current [A]")

rho_cn = pybamm.Parameter("Negative current collector density [kg.m-3]")
rho_n = pybamm.Parameters("Negative electrode density [kg.m-3]")
rho_s = pybamm.Parameter("Separator density [kg.m-3]")
rho_p = pybamm.Parameter("Positive electrode density [kg.m-3]")
rho_cp = pybamm.Parameter("Positive current collector density [kg.m-3]")

# Specific heat capacity
c_p_cn = pybamm.Parameter(
    "Negative current collector specific heat capacity [J.kg-1.K-1]"
)
c_p_n = pybamm.Parameter("Negative electrode specific heat capacity [J.kg-1.K-1]")
c_p_s = pybamm.Parameter("Separator specific heat capacity [J.kg-1.K-1]")
c_p_p = pybamm.Parameter("Negative electrode specific heat capacity [J.kg-1.K-1]")
c_p_cp = pybamm.Parameter(
    "Positive current collector specific heat capacity [J.kg-1.K-1]"
)

# Thermal conductivity
lambda_cn = pybamm.Parameter(
    "Negative current collector thermal conductivity [W.m-1.K-1]"
)
lambda_n = pybamm.Parameter("Negative electrode thermal conductivity [W.m-1.K-1]")
lambda_s = pybamm.Parameter("Separator thermal conductivity [W.m-1.K-1]")
lambda_p = pybamm.Parameter("Positive electrode thermal conductivity [W.m-1.K-1]")
lambda_cp = pybamm.Parameter(
    "Positive current collector thermal conductivity [W.m-1.K-1]"
)

# Thermal parameters
h = pybamm.Parameter("Heat transfer coefficient [W.m-2.K-1]")
Delta_T = pybamm.Parameter("Typical temperature variation [K]")
rho_eff = pybamm.Parameter("Lumped effective thermal density [J.K-1.m-3]")
lambda_eff = pybamm.Parameter("Effective thermal conductivity [W.m-1.K-1]")

# Initial temperature
T = pybamm.Parameters("Initial temperature [K]")
