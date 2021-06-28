import casadi
import numpy as np
import matplotlib.pyplot as plt

t = casadi.MX.sym("t")
x = casadi.MX.sym("x")
ode = -x

x0 = 1
t_eval = np.linspace(0, 10, 20)

sol_exact = np.exp(-t_eval)

# Casadi
opts = {"grid": t_eval, "output_t0": True}
itg = casadi.integrator("F", "cvodes", {"t": t, "x": x, "ode": ode}, opts)
sol_casadi = itg(x0=x0)["xf"].full().flatten()

# Forward Euler
ode_fn = casadi.Function("ode", [t, x], [ode])
sol_fwd = [x0]
x = x0
for i in range(len(t_eval) - 1):
    dt = t_eval[i + 1] - t_eval[i]
    step = dt * ode_fn(t_eval[i], x)
    x += step
    sol_fwd.append(x)

# Backward Euler
sol_back = [x0]
x = x0
for i in range(len(t_eval) - 1):
    dt = t_eval[i + 1] - t_eval[i]
    x = x / (1 + dt)
    sol_back.append(x)

# Crank-Nicolson
sol_CN = [x0]
x = x0
for i in range(len(t_eval) - 1):
    dt = t_eval[i + 1] - t_eval[i]
    x = (1 - dt / 2) * x / (1 + dt / 2)
    sol_CN.append(x)

plt.plot(t_eval, sol_exact, "o", label="exact")
plt.plot(t_eval, sol_casadi, label="casadi")
plt.plot(t_eval, sol_fwd, label="fwd")
plt.plot(t_eval, sol_back, label="back")
plt.plot(t_eval, sol_CN, label="CN")
plt.legend()
plt.show()
