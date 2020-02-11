#
# Constant-current constant-voltage charge
#
import pybamm
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")
experiment = pybamm.Experiment(
    [
        "Discharge at 1.5C until 2.5 V",
        "Rest for 2 hours",
    ],
    period="10 seconds",
)
model = pybamm.lithium_ion.DFN()
# param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
# param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

param = pybamm.ParameterValues(
    chemistry={
        "chemistry": "lithium-ion",
        "cell": "LGM50_Chen2020",
        "anode": "graphite_Chen2020",
        # "anode": "graphite_mcmb2528_Marquis2019",
        "separator": "separator_Chen2020",
        "cathode": "nmc_Chen2020",
        # "cathode": "lico2_Marquis2019",
        "electrolyte": "lipf6_Nyman2008",
        "experiment": "1C_discharge_from_full_Chen2020",
    }
)

cspmax = 50483 * 1.25  #1.25
csnmax = 29583 * 1.13  #1.13

param["Initial concentration in negative electrode [mol.m-3]"] = 0.90 * csnmax
param["Initial concentration in positive electrode [mol.m-3]"] = 0.26 * cspmax
# param["Initial concentration in negative electrode [mol.m-3]"] = 0.80 * csnmax
# param["Initial concentration in positive electrode [mol.m-3]"] = 0.36 * cspmax
param["Maximum concentration in negative electrode [mol.m-3]"] = csnmax
param["Maximum concentration in positive electrode [mol.m-3]"] = cspmax
param["Negative electrode Bruggeman coefficient (electrolyte)"] = 1.5
param["Positive electrode Bruggeman coefficient (electrolyte)"] = 1.5
param["Separator Bruggeman coefficient (electrolyte)"] = 1.5
param["Positive electrode diffusivity [m2.s-1]"] = 4E-15
param["Negative electrode diffusivity [m2.s-1]"] = 3.3E-14

# param["Positive electrode conductivity [S.m-1]"] = 10 # Inf detected
# param["Maximum concentration in positive electrode [mol.m-3]"] = 51217.9257309275   # Inf detected
# param["Positive electrode diffusivity [m2.s-1]"] = "[function]lico2_diffusivity_Dualfoil1998" # Convergence fail
# param["Positive electrode OCP [V]"] = "[function]lico2_ocp_Dualfoil1998" # Works
# param["Positive electrode porosity"] = 0.3 # Convergence fail
# param["Positive electrode active material volume fraction"] = 0.7 # Convergence fail
# param["Positive particle radius [m]"] = 1E-5 # Convergence fail
# param["Positive electrode surface area density [m-1]"] = 150000 # Convergence fail

# param["Negative electrode OCP [V]"] = "[function]graphite_mcmb2528_ocp_Dualfoil1998"

sim = pybamm.Simulation(
    model,
    parameter_values=param,
    experiment=experiment,
    solver=pybamm.CasadiSolver()
)
sim.solve()

# Show all plots
sim.plot()