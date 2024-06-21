#
# Compare SPMe model with and without heat of mixing
#
import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")

# load models
models = [
    pybamm.lithium_ion.SPMe(
        {"heat of mixing": "true", "thermal": "lumped"}, name="SPMe with heat of mixing"
    ),
    pybamm.lithium_ion.SPMe({"thermal": "lumped"}, name="SPMe without heat of mixing"),
]

# set parametrisation (Ecker et al., 2015)
parameter_values = pybamm.ParameterValues("Ecker2015")

# set mesh
# (increased number of points in particles to avoid oscillations for high C-rates)
var_pts = {"x_n": 16, "x_s": 8, "x_p": 16, "r_n": 128, "r_p": 128}

# set the constant current discharge C-rate
C_rate = 5

# set the simulation time and increase the number of points
# for better integration in time
t_eval = np.linspace(0, 720, 360)

# set up an extra plot with the heat of mixing vs time in each electrode and
# the integrated heat of mixing vs time in each electrode to compare with
# Figure 6(a) from Richardson et al. (2021)
fig, axs = plt.subplots(2, len(models), figsize=(12, 7))

# extract some of the parameters
L_n = parameter_values["Negative electrode thickness [m]"]
L_p = parameter_values["Positive electrode thickness [m]"]
A = parameter_values["Electrode width [m]"] * parameter_values["Electrode height [m]"]

# create and run simulations
sims = []
for m, model in enumerate(models):
    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, var_pts=var_pts, C_rate=C_rate
    )
    sim.solve(t_eval)
    sims.append(sim)

    # extract solution for plotting
    sol = sim.solution

    # extract variables from the solution
    time = sol["Time [h]"].entries
    Q_mix = sol["Heat of mixing [W.m-3]"].entries

    # heat of mixing in negative and positive electrodes multiplied by the electrode
    # width, represents the integral of heat of mixing term across each of the
    # electrodes (W.m-2)
    Q_mix_n = Q_mix[0, :] * L_n
    Q_mix_p = Q_mix[-1, :] * L_p

    # heat of mixing integrals (J.m-2)
    Q_mix_n_int = 0.0
    Q_mix_p_int = 0.0

    # data for plotting
    Q_mix_n_plt = []
    Q_mix_p_plt = []

    # performs integration in time
    for i, t in enumerate(time[1:]):
        dt = (t - time[i]) * 3600  # seconds
        Q_mix_n_avg = (Q_mix_n[i] + Q_mix_n[i + 1]) * 0.5
        Q_mix_p_avg = (Q_mix_p[i] + Q_mix_p[i + 1]) * 0.5
        # convert J to kJ and divide the integral by the electrode area A to compare
        # with Figure 6(a) from Richardson et al. (2021)
        Q_mix_n_int += Q_mix_n_avg * dt / 1000 / A
        Q_mix_p_int += Q_mix_p_avg * dt / 1000 / A
        Q_mix_n_plt.append(Q_mix_n_int)
        Q_mix_p_plt.append(Q_mix_p_int)

    # plots heat of mixing in each electrode vs time in minutes
    axs[0, m].plot(time * 60, Q_mix_n, ls="-", label="Negative electrode")
    axs[0, m].plot(time * 60, Q_mix_p, ls="--", label="Positive electrode")
    axs[0, m].set_title(f"{model.name}")
    axs[0, m].set_xlabel("Time [min]")
    axs[0, m].set_ylabel("Heat of mixing [W.m-2]")
    axs[0, m].grid(True)
    axs[0, m].legend()

    # plots integrated heat of mixing in each electrode vs time in minutes
    axs[1, m].plot(time[1:] * 60, Q_mix_n_plt, ls="-", label="Negative electrode")
    axs[1, m].plot(time[1:] * 60, Q_mix_p_plt, ls="--", label="Positive electrode")
    axs[1, m].set_xlabel("Time [min]")
    axs[1, m].set_ylabel("Integrated heat of mixing [kJ.m-2]")
    axs[1, m].grid(True)
    axs[1, m].legend()

# plot
pybamm.dynamic_plot(
    sims,
    output_variables=[
        "X-averaged cell temperature [K]",
        "X-averaged heat of mixing [W.m-3]",
        "X-averaged total heating [W.m-3]",
        "Heat of mixing [W.m-3]",
        "Voltage [V]",
        "Current [A]",
    ],
)
