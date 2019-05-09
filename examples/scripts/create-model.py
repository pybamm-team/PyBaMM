# This script is intended to be a stripped back version of the
# 'examples/notebooks/create-model.ipnb' so for more details please see
# that notebook

import pybamm

# 1. Initialise model ------------------------------------------------------------------
model = pybamm.BaseModel()

# 2. Define parameters and variables ---------------------------------------------------
# dimensional parameters
k_dim = pybamm.Parameter("Reaction rate constant")
L_0_dim = pybamm.Parameter("Initial thickness")
V_hat_dim = pybamm.Parameter("Partial molar volume")
c_inf_dim = pybamm.Parameter("Bulk electrolyte solvent concentration")


def D_dim(cc):
    return pybamm.FunctionParameter("Diffusivity", cc)


# dimensionless parameters
k = k_dim * L_0_dim / D_dim(c_inf_dim)
V_hat = V_hat_dim * c_inf_dim


def D(cc):
    c_dim = c_inf_dim * cc
    return D_dim(c_dim) / D_dim(c_inf_dim)


# Define variables
c = pybamm.Variable("Solvent concentration")  # this probably needs a domain
L = pybamm.Variable("SEI thickness")

# 3. State governing equations ---------------------------------------------------------
R = -k * pybamm.BoundaryValue(c, "left")  # SEI reaction flux
N = -(1 / L) * D(c) * pybamm.grad(c)  # solvent flux
dcdt = -(1 / L) * pybamm.div(N)  # solvent concentration governing equation
dLdt = -V_hat * R  # SEI thickness governing equation

model.rhs = {c: dcdt, L: dLdt}  # add to model

# 4. State boundary conditions ---------------------------------------------------------
grad_c_left = L * R / D(pybamm.BoundaryValue(c, "left"))  # left bc
c_right = pybamm.Scalar(1)  # right bc

# add to model
model.boundary_conditions = {
    c: {"left": (grad_c_left, "Neumann"), "right": (c_right, "Dirichlet")}
}

# 5. State initial conditions ----------------------------------------------------------
model.initial_conditions = {c: pybamm.Scalar(1), L: pybamm.Scalar(1)}

# 6. State output variables ------------------------------------------------------------
model.variables = {
    "SEI thickness": L,
    "SEI growth rate": dLdt,
    "Solvent concentration": c,
    "SEI thickness [m]": L_0_dim * L,
    "SEI growth rate [m/s]": (D_dim(c_inf_dim) / L_0_dim) * dLdt,
    "Solvent concentration [mols/m^3]": c_inf_dim * c,
}
