from casadi import *
import time

# Define ode
t = MX.sym("t")
x = MX.sym("x")
p = MX.sym("p")

x0 = np.ones(x.shape[0])
t_eval = np.linspace(0, 1, 100)

t_max = MX.sym("t_min")
t_min = MX.sym("t_max")
tlims = casadi.vertcat(t_min, t_max)

ode = 1  # -(t_max - t_min) * p * x

# value of the parameter for evaluating
p_eval = 1

# First approach: simple integrator without a grid
# fastest but doesn't give the intermediate points
print("no grid")
print("*" * 10)
itg_nogrid = integrator(
    "F",
    "cvodes",
    {"t": t, "x": x, "ode": ode, "p": casadi.vertcat(p, tlims), "quad": 1},
)

start = time.time()
itg_nogrid(x0=1, p=[p_eval, 0, 1])
print("value:", time.time() - start)

jac_nogrid = Function(
    "j", [p], [jacobian(itg_nogrid(x0=x0, p=casadi.vertcat(p, 0, 1))["xf"], p)]
)
start = time.time()
jac_nogrid(1)
print("jacobian:", time.time() - start)

# # Second approach: integrator with a grid
# # fast, gives intermediate points, but can't take the jacobian
# print("With grid")
# print("*" * 10)
# itg_grid_auto = integrator(
#     "F",
#     "cvodes",
#     {"t": t, "x": x, "ode": ode, "p": casadi.vertcat(p, tlims)},
#     {"grid": t_eval, "output_t0": True},
# )

# start = time.time()
# itg_grid_auto(x0=1, p=[p_eval, 0, 1])
# print("value:", time.time() - start)

# # Fails
# # jac_grid_auto = Function(
# #     "j", [p], [jacobian(itg_grid_auto(x0=x0, p=casadi.vertcat(p, 0, 1))["xf"], p)]
# # )
# # jac_grid_auto(1)
# print("jacobian: fails")

# # Third approach: multiple calls through manual for loop
# print("For loop")
# print("*" * 10)


# def itg_grid_manual(x0, p_eval, t_eval):
#     X = x0
#     for i in range(t_eval.shape[0] - 1):
#         t_min = t_eval[i]
#         t_max = t_eval[i + 1]
#         xnew = itg_nogrid(x0=x0, p=casadi.vertcat(p_eval, t_min, t_max))["xf"]
#         X = casadi.horzcat(X, xnew)
#         x0 = xnew
#     return X


# start = time.time()
# itg_grid_manual(x0, p_eval, t_eval)
# print("value:", time.time() - start)

# jac_grid_manual = Function("j", [p], [jacobian(itg_grid_manual(x0, p, t_eval), p)],)
# start = time.time()
# jac_grid_manual(1)
# print("jacobian:", time.time() - start)


# # Fourth approach: multiple calls through mapaccum
# print("mapaccum")
# print("*" * 10)

# x0_sym = MX.sym("x0", x.shape[0])
# itg_fn = Function(
#     "itg_fn",
#     [x0_sym, p, tlims],
#     [itg_nogrid(x0=x0_sym, p=casadi.vertcat(p, tlims))["xf"]],
# )
# itg_grid_mapaccum = itg_fn.mapaccum("Fn", len(t_eval) - 1)

# tlims_eval = casadi.horzcat(t_eval[:-1], t_eval[1:]).T

# start = time.time()
# itg_grid_mapaccum(x0, p_eval, tlims_eval)
# print("value:", time.time() - start)

# # Jacobians
# jac_grid_mapaccum = Function(
#     "j", [p], [jacobian(itg_grid_mapaccum(x0, p, tlims_eval), p)]
# )
# start = time.time()
# jac_grid_mapaccum(1)
# print("jacobian:", time.time() - start)
