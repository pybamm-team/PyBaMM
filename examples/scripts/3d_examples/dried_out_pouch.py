import numpy as np
import pybamm

model = pybamm.lithium_ion.DFN(
    {"current collector": "potential pair", "dimensionality": 2}
)
param = pybamm.ParameterValues("Ecker2015")
Ly = param["Electrode width [m]"]
Lz = param["Electrode height [m]"]


def _sigmoid(arg):
    return (1 + np.tanh(arg)) / 2


def _top_hat(arg, a, b, k=500):
    return _sigmoid(k * (arg - a)) * _sigmoid(k * (b - arg))


# Simulate drying out of the negative electrode edges by reducing porosity
def eps_n(x, y, z):
    return 0.329 * (
        _top_hat(arg=y, a=Ly * 0.02, b=Ly * 0.98)
        * _top_hat(arg=z, a=Lz * 0.02, b=Lz * 0.98)
    )


param.update({"Negative electrode porosity": eps_n})
var_pts = {"x_n": 8, "x_s": 8, "x_p": 8, "r_n": 8, "r_p": 8, "y": 24, "z": 24}
exp = pybamm.Experiment(
    [
        "Discharge at 1C until 2.7 V",
        "Charge at 1C until 4.2 V",
        "Hold at 4.2 V until C/20",
    ]
)
sim = pybamm.Simulation(model, var_pts=var_pts, parameter_values=param, experiment=exp)
sol = sim.solve()
output_variables = [
    "X-averaged negative electrode porosity",
    "X-averaged negative particle surface stoichiometry",
    "X-averaged negative electrode surface potential difference [V]",
    "Current collector current density",
    "Current [A]",
    "Voltage [V]",
]
plot = sol.plot(output_variables, variable_limits="tight", shading="auto")
