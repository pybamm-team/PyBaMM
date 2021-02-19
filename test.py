import pybamm
import matplotlib.pyplot as plt
import numpy as np

parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
parameter_values.update(
    {
        "SEI kinetic rate constant [m.s-1]": 0,  # 1e-20,
        "SEI resistivity [Ohm.m]": 0,
    }
)
spm = pybamm.lithium_ion.SPM()  # {"sei": "ec reaction limited"})

esoh_model = pybamm.lithium_ion.ElectrodeSOH()
esoh_sim = pybamm.Simulation(esoh_model, parameter_values=parameter_values)
param = spm.param

Vmin = 2.5
Vmax = 4.2
Cn = parameter_values.evaluate(param.C_n_init)
Cp = parameter_values.evaluate(param.C_p_init)
n_Li_init = parameter_values.evaluate(param.n_Li_particles_init)
c_n_max = parameter_values.evaluate(param.c_n_max)
c_p_max = parameter_values.evaluate(param.c_p_max)

inputs = {"V_min": Vmin, "V_max": Vmax, "C_n": Cn, "C_p": Cp, "n_Li": n_Li_init}
print(inputs)

esoh_sol = esoh_sim.solve(
    [0],
    inputs=inputs,
)
for x in ["x_0", "x_100", "y_0", "y_100"]:
    print(x, ":", esoh_sol[x].data[0])

parameter_values.update(
    {
        "Initial concentration in negative electrode [mol.m-3]": esoh_sol["x_100"].data[
            0
        ]
        * c_n_max,
        "Initial concentration in positive electrode [mol.m-3]": esoh_sol["y_100"].data[
            0
        ]
        * c_p_max,
    }
)
experiment = pybamm.Experiment(
    [
        (
            f"Discharge at 1C until {Vmin}V",
            "Rest for 1 hour",
            f"Charge at 1C until {Vmax}V",
            f"Hold at {Vmax}V until C/50",
        )
    ]
)
pybamm.set_logging_level("NOTICE")
sim = pybamm.Simulation(spm, experiment=experiment, parameter_values=parameter_values)
spm_sol = sim.solve()