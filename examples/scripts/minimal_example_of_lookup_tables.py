import pybamm
import pandas as pd
import numpy as np


def process_2D(name, data):
    data = data.to_numpy()
    x1 = np.unique(data[:, 0])
    x2 = np.unique(data[:, 1])

    value = data[:, 2]

    x = (x1, x2)

    value_data = value.reshape(len(x1), len(x2), order="C")

    formatted_data = (name, (x, value_data))

    return formatted_data


parameter_values = pybamm.ParameterValues(pybamm.parameter_sets.Chen2020)

# overwrite the diffusion coefficient with a 2D lookup table
D_s_n = parameter_values["Negative electrode diffusivity [m2.s-1]"]
df = pd.DataFrame(
    {
        "T": [0, 0, 25, 25, 45, 45],
        "sto": [0, 1, 0, 1, 0, 1],
        "D_s_n": [D_s_n, D_s_n, D_s_n, D_s_n, D_s_n, D_s_n],
    }
)
df["T"] = df["T"] + 273.15
D_s_n_data = process_2D("Negative electrode diffusivity [m2.s-1]", df)


def D_s_n(sto, T):
    name, (x, y) = D_s_n_data
    return pybamm.Interpolant(x, y, [T, sto], name)


parameter_values["Negative electrode diffusivity [m2.s-1]"] = D_s_n

k_n = parameter_values["Negative electrode exchange-current density [A.m-2]"]

model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, parameter_values=parameter_values)

sim.solve([0, 30])

sim.plot(["Negative particle surface concentration [mol.m-3]"])
