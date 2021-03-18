import pybamm

model = pybamm.lithium_ion.SPMe(name="SPM")
sim = pybamm.Simulation(model)
sim.set_parameters()

model = sim.model

mtk_str = pybamm.get_julia_mtk_model(model, geometry=sim.geometry, tspan=(0, 3600))
print(mtk_str)

# sim.build()

# rhs_str, u0_str = sim.built_model.generate_julia_diffeq()
# print(rhs_str)