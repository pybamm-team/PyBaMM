import pybamm
import numpy as np

model = pybamm.lithium_ion.DFN(
    {"current collector": "potential pair", "dimensionality": 1}
)
param = pybamm.ParameterValues("Ecker2015")
Ly = param["Electrode width [m]"]
Lz = param["Electrode height [m]"]

def sigmoid(arg):
    return (1 + np.tanh(arg)) / 2


def top_hat(arg, a, b, k=500):
    return sigmoid(k * (arg - a)) * sigmoid(k * (b - arg))


def eps_s_n(x, y_cc, z_cc):
    return 0.372403 * (
        top_hat(arg=z_cc, a=Lz*0.05, b=Lz*0.95)
    )


def eps_s_p(x, y_cc, z_cc):
    return 0.40832 * (
        top_hat(arg=z_cc, a=Lz*0.05, b=Lz*0.95)
    )

param_dryout = param.copy()
param_dryout.update(
    {
        "Negative electrode active material volume fraction": eps_s_n,
        "Positive electrode active material volume fraction": eps_s_p,
    }
)

var_pts = {"x_n": 8, "x_s": 8, "x_p": 8, "r_n": 8, "r_p": 8, "z": 32}
exp = pybamm.Experiment(
    [
        "Discharge at 1C until 2.7 V",
        "Charge at 1C until 4.2 V",
        "Hold at 4.2 V until C/20"
    ]
)
sim = pybamm.Simulation(
    model, var_pts=var_pts, parameter_values=param_dryout, experiment=exp
)
sol = sim.solve()
output_variables = [
    "X-averaged negative electrode active material volume fraction",
    "X-averaged positive electrode active material volume fraction",
    "Current collector current density [A.m-2]",
    "X-averaged negative particle surface stoichiometry",
    "X-averaged negative electrode surface potential difference [V]",
    "Voltage [V]",
]
plot = sol.plot(output_variables, variable_limits="tight", shading="auto")
