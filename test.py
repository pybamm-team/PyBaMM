import pybamm
import numpy as np


def graphite_LGM50_ocp_Chen2020(sto):
    u_eq = (
        1.9793 * np.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * np.tanh(30.4444 * (sto - 0.6103))
    )

    return u_eq


def nmc_LGM50_ocp_Chen2020(sto):
    u_eq = (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (sto - 0.3120))
    )

    return u_eq


# state variables
x_n = pybamm.Variable("Negative electrode sto")
x_p = pybamm.Variable("Positive electrode sto")

# parameters
x_n_0 = pybamm.Parameter("Negative electrode initial sto")
x_p_0 = pybamm.Parameter("Positive electrode initial sto")
i = pybamm.FunctionParameter("Current [A]", {"Time [s]": pybamm.t})
u_n = pybamm.FunctionParameter("Negative electrode potential [V]", {"x_n": x_n})
u_p = pybamm.FunctionParameter("Positive electrode potential [V]", {"x_p": x_p})
q_n = pybamm.Parameter("Negative electrode capacity [A.h]")
q_p = pybamm.Parameter("Positive electrode capacity [A.h]")
r = pybamm.Parameter("Resistance [Ohm]")

# pybamm model
model = pybamm.BaseModel("Simple reservoir model")
model.rhs[x_n] = -i / q_n
model.rhs[x_p] = i / q_p
model.initial_conditions[x_n] = x_n_0
model.initial_conditions[x_p] = x_p_0
model.variables["Negative electrode sto"] = x_n
model.variables["Positive electrode sto"] = x_p

# events


# parameter values
parameter_values = pybamm.ParameterValues(
    {
        "Negative electrode initial sto": 0.9,
        "Positive electrode initial sto": 0.1,
        "Negative electrode capacity [A.h]": 1,
        "Positive electrode capacity [A.h]": 1,
        "Resistance [Ohm]": 1,
        "Current [A]": lambda t: 1 + 0.5 * pybamm.sin(100 * t),
        "Negative electrode potential [V]": graphite_LGM50_ocp_Chen2020,
        "Positive electrode potential [V]": nmc_LGM50_ocp_Chen2020,
    }
)

# solver
sim = pybamm.Simulation(model, parameter_values=parameter_values)
sol = sim.solve([0, 3600])
sol.plot(["Negative electrode sto", "Positive electrode sto"])
