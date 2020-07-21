import pybamm
import casadi

model = pybamm.lithium_ion.SPMe()
param = model.default_parameter_values
param["Negative electrode porosity"] = 0.3
param["Separator porosity"] = 0.3
param["Positive electrode porosity"] = 0.3
param["Cation transference number"] = pybamm.InputParameter("t")

solver = pybamm.CasadiSolver(mode="fast")  # , sensitivity=True)
sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
sol = sim.solve([0, 3600], inputs={"t": 0.5})

# print(sol["X-averaged electrolyte concentration"].data)
var = sol["Terminal voltage [V]"]

t = casadi.MX.sym("t")
y = casadi.MX.sym("y", sim.built_model.len_rhs)
p = casadi.MX.sym("p")

rhs = sim.built_model.casadi_rhs(t, y, p)

jac_x_func = casadi.Function("jac_x", [t, y, p], [casadi.jacobian(rhs, y)])
jac_p_func = casadi.Function("jac_x", [t, y, p], [casadi.jacobian(rhs, p)])
for idx in range(len(sol.t)):
    t = sol.t[idx]
    u = sol.y[:, idx]
    inp = 0.5
    next_jac_x_eval = jac_x_func(t, u, inp)
    next_jac_p_eval = jac_p_func(t, u, inp)
    if idx == 0:
        jac_x_eval = next_jac_x_eval
        jac_p_eval = next_jac_p_eval
    else:
        jac_x_eval = casadi.diagcat(jac_x_eval, next_jac_x_eval)
        jac_p_eval = casadi.diagcat(jac_p_eval, next_jac_p_eval)

# Convert variable to casadi format for differentiating
# var_casadi = self.base_variable.to_casadi(t_casadi, y_casadi, inputs=p_casadi)
# dvar_dy = casadi.jacobian(var_casadi, y_casadi)
# dvar_dp = casadi.jacobian(var_casadi, p_casadi_stacked)

# # Convert to functions and evaluate index-by-index
# dvar_dy_func = casadi.Function(
#     "dvar_dy", [t_casadi, y_casadi, p_casadi_stacked], [dvar_dy]
# )
# dvar_dp_func = casadi.Function(
#     "dvar_dp", [t_casadi, y_casadi, p_casadi_stacked], [dvar_dp]
# )
# for idx in range(len(self.t_sol)):
#     t = self.t_sol[idx]
#     u = self.u_sol[:, idx]
#     inp = inputs_stacked[:, idx]
#     next_dvar_dy_eval = dvar_dy_func(t, u, inp)
#     next_dvar_dp_eval = dvar_dp_func(t, u, inp)
#     if idx == 0:
#         dvar_dy_eval = next_dvar_dy_eval
#         dvar_dp_eval = next_dvar_dp_eval
#     else:
#         dvar_dy_eval = casadi.diagcat(dvar_dy_eval, next_dvar_dy_eval)
#         dvar_dp_eval = casadi.vertcat(dvar_dp_eval, next_dvar_dp_eval)

