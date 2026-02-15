import matplotlib.pyplot as plt

import pybamm

model = pybamm.lithium_ion.MSMR({"number of MSMR reactions": ("6", "4")})
parameter_values = model.default_parameter_values
parameter_values["Positive electrode host site occupancy fraction (0)"] = "[input]"
parameter_values["Positive electrode host site standard potential (0) [V]"] = 3.62274
parameter_values["Positive electrode host site ideality factor (0)"] = 0.9671
parameter_values["Positive electrode host site charge transfer coefficient (0)"] = 0.5
parameter_values[
    "Positive electrode host site reference exchange-current density (0) [A.m-2]"
] = 5
parameter_values["Negative electrode host site occupancy fraction (0)"] = 0.43336
parameter_values["Negative electrode host site standard potential (0) [V]"] = 0.08843

# Loop over domains
for domain in ["negative", "positive"]:
    Electrode = domain.capitalize()
    # Loop over reactions
    N = int(parameter_values["Number of reactions in " + domain + " electrode"])
    for i in range(N):
        names = [
            f"{Electrode} electrode host site occupancy fraction ({i})",
            f"{Electrode} electrode host site standard potential ({i}) [V]",
            f"{Electrode} electrode host site ideality factor ({i})",
            f"{Electrode} electrode host site charge transfer coefficient ({i})",
            f"{Electrode} electrode host site reference exchange-current density ({i}) [A.m-2]",
        ]
        for name in names:
            print(f"{name} = {parameter_values[name]}")

# get symbolic parameters
param = model.param
param_n = param.n.prim
param_p = param.p.prim

num_reactions_n = int(parameter_values["Number of reactions in negative electrode"])
num_reactions_p = int(parameter_values["Number of reactions in positive electrode"])

# set up ranges for plotting
U_n = pybamm.linspace(0.05, 1.1, 1000)
U_p = pybamm.linspace(2.8, 4.4, 1000)

# get reference electrolyte concentration and temperature
c_e = param.c_e_init
T = param.T_init

# set up figure
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
colors = ["r", "g", "b", "c", "m", "y"]

# sto vs potential
x_n = param_n.x(U_n, T)
x_p = param_p.x(U_p, T)
ax[0, 0].plot(parameter_values.evaluate(x_n), parameter_values.evaluate(U_n), "k-")
ax[0, 1].plot(parameter_values.evaluate(x_p), parameter_values.evaluate(U_p), "k-")
ax[0, 0].set_xlabel("x_n")
ax[0, 0].set_ylabel("U_n [V]")
ax[0, 1].set_xlabel("x_p")
ax[0, 1].set_ylabel("U_p [V]")

# fractional occupancy vs potential
for i in range(num_reactions_n):
    xj = param_n.x_j(U_n, T, i)
    ax[1, 0].plot(
        parameter_values.evaluate(x_n),
        parameter_values.evaluate(xj),
        color=colors[i],
        label=f"x_n_{i}",
    )
ax[1, 0].set_xlabel("x_n")
ax[1, 0].set_ylabel("x_n_j")
ax[1, 0].legend()
for i in range(num_reactions_p):
    xj = param_p.x_j(U_p, T, i)
    ax[1, 1].plot(
        parameter_values.evaluate(x_p),
        parameter_values.evaluate(xj),
        color=colors[i],
        label=f"x_p_{i}",
    )
ax[1, 1].set_xlabel("x_p")
ax[1, 1].set_ylabel("x_p_j")
ax[1, 1].legend()

# exchange current density vs potential
for i in range(num_reactions_n):
    xj = param_n.x_j(U_n, T, i)
    j0 = param_n.j0_j(c_e, U_n, T, i)
    ax[2, 0].plot(
        parameter_values.evaluate(x_n),
        parameter_values.evaluate(j0),
        color=colors[i],
        label=f"j0_n_{i}",
    )
ax[2, 0].set_xlabel("x_n")
ax[2, 0].set_ylabel("j0_n_j [A.m-2]")
ax[2, 0].legend()
for i in range(num_reactions_p):
    xj = param_p.x_j(U_p, T, i)
    j0 = param_p.j0_j(c_e, U_p, T, i)
    ax[2, 1].plot(
        parameter_values.evaluate(x_p),
        parameter_values.evaluate(j0),
        color=colors[i],
        label=f"j0_p_{i}",
    )
ax[2, 1].set_ylim([0, 0.5])
ax[2, 1].set_xlabel("x_p")
ax[2, 1].set_ylabel("j0_p_j [A.m-2]")
ax[2, 1].legend()

plt.tight_layout()
experiment = pybamm.Experiment(
    [
        (
            "Discharge at 1C for 1 hour or until 3 V",
            "Rest for 1 hour",
            "Charge at C/3 until 4.2 V",
            "Hold at 4.2 V until 10 mA",
            "Rest for 1 hour",
        ),
    ],
)
sim = pybamm.Simulation(model, experiment=experiment)
sim.solve(inputs={"Positive electrode host site occupancy fraction (0)": 0.14442})
sim.plot(
    [
        "Negative particle stoichiometry",
        "Positive particle stoichiometry",
        "X-averaged negative electrode open-circuit potential [V]",
        "X-averaged positive electrode open-circuit potential [V]",
        "Negative particle potential [V]",
        "Positive particle potential [V]",
        "Current [A]",
        "Voltage [V]",
    ],
    variable_limits="tight",  # make axes tight to plot at each timestep
)
