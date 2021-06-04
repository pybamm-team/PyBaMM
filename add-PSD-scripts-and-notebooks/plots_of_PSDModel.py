import pybamm
import matplotlib.pyplot as plt


pybamm.set_logging_level("DEBUG")

# Experiment
# (Use default 1C discharge from full)
t_eval = [0, 3600]

# Parameter values
params = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
R_n_dim = params["Negative particle radius [m]"]
R_p_dim = params["Positive particle radius [m]"]

# Add distribution parameters

# Standard deviations
sd_a_n = 0.3  # dimensionless
sd_a_p = 0.3
sd_a_n_dim = sd_a_n * R_n_dim  # dimensional
sd_a_p_dim = sd_a_p * R_p_dim

# Minimum and maximum particle sizes (dimensionaless)
R_min_n = 0
R_min_p = 0
R_max_n = max(2, 1 + sd_a_n * 5)
R_max_p = max(2, 1 + sd_a_p * 5)


def lognormal_distribution(R, R_av, sd):
    '''
    A lognormal distribution with arguments
        R :     particle radius
        R_av:   mean particle radius
        sd :    standard deviation
    '''
    import numpy as np

    # calculate usual lognormal parameters
    mu_ln = pybamm.log(R_av ** 2 / pybamm.sqrt(R_av ** 2 + sd ** 2))
    sigma_ln = pybamm.sqrt(pybamm.log(1 + sd ** 2 / R_av ** 2))
    return (
        pybamm.exp(-((pybamm.log(R) - mu_ln) ** 2) / (2 * sigma_ln ** 2))
        / pybamm.sqrt(2 * np.pi * sigma_ln ** 2)
        / R
    )


# Set the dimensional (area-weighted) particle-size distributions
# Note: the only argument must be the particle size R
def f_a_dist_n_dim(R):
    return lognormal_distribution(R, R_n_dim, sd_a_n_dim)


def f_a_dist_p_dim(R):
    return lognormal_distribution(R, R_p_dim, sd_a_p_dim)


# input distribution params (dimensional)
distribution_params = {
    "Negative area-weighted particle-size "
    + "standard deviation [m]": sd_a_n_dim,
    "Positive area-weighted particle-size "
    + "standard deviation [m]": sd_a_p_dim,
    "Negative minimum particle radius [m]": R_min_n * R_n_dim,
    "Positive minimum particle radius [m]": R_min_p * R_p_dim,
    "Negative maximum particle radius [m]": R_max_n * R_n_dim,
    "Positive maximum particle radius [m]": R_max_p * R_p_dim,
    "Negative area-weighted "
    + "particle-size distribution [m-1]": f_a_dist_n_dim,
    "Positive area-weighted "
    + "particle-size distribution [m-1]": f_a_dist_p_dim,
}
params.update(distribution_params, check_already_exists=False)

# MPM
model_1 = pybamm.lithium_ion.MPM(name="MPM")  # default params

# DFN with PSD option
model_2 = pybamm.lithium_ion.DFN(
    #options={"particle-size distribution": "true"},
    #name="MP-DFN"
)

# DFN (no particle-size distributions)
model_3 = pybamm.lithium_ion.DFN(name="DFN")

models = [model_1, model_2, model_3]

sims=[]
for model in models:
    sim = pybamm.Simulation(
        model,
        parameter_values=params,
        solver=pybamm.CasadiSolver(mode="fast")
    )
    sims.append(sim)

# Reduce number of points in R
var = pybamm.standard_spatial_vars
sims[1].var_pts.update(
    {
        var.R_n: 20,
        var.R_p: 20,
    }
)

# Solve
for sim in sims:
    sim.solve(t_eval=t_eval)


# Plot
output_variables = [
    "Negative particle surface concentration",
    "Positive particle surface concentration",
    "X-averaged negative particle surface concentration distribution",
    "X-averaged positive particle surface concentration distribution",
#    "Negative particle surface concentration distribution",
#    "Positive particle surface concentration distribution",
    "Negative area-weighted particle-size distribution",
    "Positive area-weighted particle-size distribution",
    "Terminal voltage [V]",
]
# MPM
sims[0].plot(output_variables)
# MPM and MP-DFN
#pybamm.dynamic_plot([sims[0], sims[1]], output_variables=output_variables)
#pybamm.dynamic_plot(sims[1], output_variables=[
#    "Negative particle surface concentration distribution",
#    "Positive particle surface concentration distribution",
#])
# MPM, MP-DFN and DFN
pybamm.dynamic_plot(sims)
