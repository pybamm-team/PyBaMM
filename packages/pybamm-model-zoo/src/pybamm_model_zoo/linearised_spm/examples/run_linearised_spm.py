"""Fit a diffusivity to a GITT pulse, the way the Weppner-Huggins method does."""

import numpy as np

import pybamm
import pybamm_model_zoo as zoo

FARADAY = 96485.33212
DIFFUSIVITY = 1e-14

model = zoo.load("LinearisedSPM")({"working electrode": "positive"})
parameter_values = model.default_parameter_values
parameter_values["Positive particle diffusivity [m2.s-1]"] = DIFFUSIVITY
radius = parameter_values["Positive particle radius [m]"]
concentration_max = parameter_values[
    "Maximum concentration in positive electrode [mol.m-3]"
]
simulation = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    var_pts={**model.default_var_pts, "r_p": 200},
)

print(f"particle diffusion time R^2/D = {radius**2 / DIFFUSIVITY:.0f} s")
print("  pulse [s]   dV/dsqrt(t) [V.s-0.5]   D fitted [m2.s-1]   error")
for pulse in (80.0, 20.0, 5.0):
    solution = simulation.solve([0, pulse])
    times = np.linspace(pulse / 20, pulse, 60)
    gradient = solution["Positive electrode open-circuit potential gradient [V]"](
        times[0]
    )
    flux = (
        solution["X-averaged positive electrode interfacial current density [A.m-2]"](
            times[0]
        )
        / FARADAY
    )
    slope = np.polyfit(np.sqrt(times), solution["Voltage [V]"](times), 1)[0]
    fitted = (4 / np.pi) * (gradient * flux / (concentration_max * slope)) ** 2
    print(
        f"{pulse:11.0f}   {slope:21.3e}   {fitted:17.3e}   "
        f"{fitted / DIFFUSIVITY - 1:+6.1%}"
    )
