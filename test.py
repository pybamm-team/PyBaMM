import pybamm

model = pybamm.lithium_ion.DFN({"particle": "uniform profile"}, name="DFN_no_r")
# model = pybamm.BaseModel(name="DFN_no_r")
# var1 = pybamm.Variable("var1")
# var2 = pybamm.Variable("var2")
# model.rhs = {var1: 0.1 * var1}
# model.algebraic = {var2: 2 * var1 - var2}
# model.initial_conditions = {var1: 1, var2: 2}
# parameter_values = model.default_parameter_values
# parameter_values._replace_callable_function_parameters = False
sim = pybamm.Simulation(model)
# sim.set_parameters()
# mtk_str = pybamm.get_julia_mtk_model(sim.model, geometry=sim.geometry, tspan=(0, 3600))


# print(mtk_str)
# list(sim.model.rhs.values())[3].render()

sim.build()

rhs_str, u0_str = sim.built_model.generate_julia_diffeq(
    get_consistent_ics_solver=pybamm.CasadiSolver()
)
print(rhs_str)
print(u0_str)