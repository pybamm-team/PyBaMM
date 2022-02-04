import pybamm

model = pybamm.lithium_ion.SPM(name="SPM")
# model = pybamm.BaseModel(name="DFN_no_r")
# var1 = pybamm.Variable("var1")
# var2 = pybamm.Variable("var2")
# model.rhs = {var1: 0.1 * var1}
# model.algebraic = {var2: 2 * var1 - var2}
# model.initial_conditions = {var1: 1, var2: 2}
parameter_values = model.default_parameter_values
# parameter_values["Electrolyte diffusivity [m2.s-1]"] = 1e-10
# parameter_values["Electrolyte conductivity [S.m-1]"] = 1
# parameter_values["Negative electrode exchange-current density [A.m-2]"] = 1e-6
# parameter_values["Positive electrode exchange-current density [A.m-2]"] = 1e-6
# parameter_values["Negative electrode OCP [V]"] = 0.5
# parameter_values["Positive electrode OCP [V]"] = 4
# parameter_values._replace_callable_function_parameters = True

var = pybamm.standard_spatial_vars
var_pts = {"x_n": 10, "x_s": 10, "x_p": 10, "r_n": 10, "r_p": 10}

sim = pybamm.Simulation(model, var_pts=var_pts, parameter_values=parameter_values)
# sim.set_parameters()
# mtk_str = pybamm.get_julia_mtk_model(sim.model, geometry=sim.geometry, tspan=(0, 3600))


# print(mtk_str)
# # list(sim.model.rhs.values())[3].render()

sim.build()

rhs_str, u0_str = sim.built_model.generate_julia_diffeq(
    get_consistent_ics_solver=pybamm.CasadiSolver(), preallocate=False
)
print(rhs_str)
print(u0_str)

V_str = pybamm.get_julia_function(
    sim.built_model.variables["Terminal voltage [V]"], funcname="V"
)
print(V_str)
