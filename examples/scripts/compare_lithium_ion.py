#
# Compare lithium-ion battery models
#
import pybamm

# Chen 2020 plating: pos = function, neg = data
param = pybamm.ParameterValues("Chen2020_plating")
model = pybamm.lithium_ion.ElectrodeSOH()
spm = pybamm.lithium_ion.SPM()
sim = pybamm.Simulation(model, parameter_values=param)
sim.build()

Cn = param.evaluate(spm.param.C_n_init)
Cp = param.evaluate(spm.param.C_p_init)
nLi = param.evaluate(spm.param.n_Li_init)
V_max = 4.2
V_min = 2.8
inputs = {"C_n": Cn, "C_p": Cp, "n_Li": nLi, "V_max": V_max, "V_min": V_min}

y_init = sim.built_model.concatenated_initial_conditions.evaluate(inputs=inputs)
y_sol = sim.solve([0], inputs=inputs).y

print(sim.built_model.concatenated_algebraic.evaluate(t=0, y=y_init, inputs=inputs))
print(sim.built_model.concatenated_algebraic.evaluate(t=0, y=y_sol, inputs=inputs))
