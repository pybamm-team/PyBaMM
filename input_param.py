import pybamm
from pybamm import exp, constants

model = pybamm.lithium_ion.SPM()

param = model.default_parameter_values


def diffusivity(sto, T):
    D_ref = 3.9 * 10 ** (-14)
    # E_D_s = 42770
    E_D_s = pybamm.Parameter("Negative solid diffusion activation energy [J.mol-1]")
    arrhenius = exp(E_D_s / constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


param["Negative electrode diffusivity [m2.s-1]"] = diffusivity

param.update(
    {"Negative solid diffusion activation energy [J.mol-1]": "[input]"},
    check_already_exists=False,
)

sim = pybamm.Simulation(model, parameter_values=param)
sim.solve(inputs={"Negative solid diffusion activation energy [J.mol-1]": 42000})
sim.plot()
