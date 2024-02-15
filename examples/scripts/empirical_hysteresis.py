#
# Empirical hysteresis modelling
#
import pybamm
from numpy import array

pybamm.set_logging_level("INFO")

# Load model
options = {
    "open-circuit potential": ("current sigmoid", "single"),
    "exchange-current density": ("current sigmoid", "single"),
    "diffusivity": ("current sigmoid", "single"),
}
model = pybamm.lithium_ion.SPMe(options)

# Load parameter values and add (de)lithiation OCPs, exchange-current densities,
# and diffusion coefficients for the negative electrode. Note: these are only intended
# to be illustrative, and are not based on any particular data.
parameter_values = pybamm.ParameterValues("Chen2020")


def ocp_lithiation(sto):
    R = 8.3145
    T = 298.15
    F = 96485
    p = array(
        [
            5.03704200e02,
            -8.82372514e-06,
            4.56628658e02,
            5.61890927e02,
            3.87812964e00,
            -4.02931829e00,
            1.52206158e03,
            -1.59630048e01,
            -2.67563363e-01,
            5.20566396e-02,
            -6.09073875e00,
            2.65427032e-01,
            1.46011356e-02,
            -1.64753973e01,
            7.39882291e-01,
        ]
    )
    u_eq = (
        p[0] * pybamm.exp(-p[1] * sto)
        + p[2]
        + p[3] * pybamm.tanh(p[4] * (sto - p[5]))
        + p[6] * pybamm.tanh(p[7] * (sto - p[8]))
        + p[9] * pybamm.tanh(p[10] * (sto - p[11]))
        + p[12] * pybamm.tanh(p[13] * (sto - p[14]))
        - 0.5 * R * T / F * pybamm.log(sto)
    )
    return u_eq


def ocp_delithiation(sto):
    R = 8.3145
    T = 298.15
    F = 96485
    p = array(
        [
            3.91897019e01,
            2.55972942e00,
            -3.12343803e01,
            2.75190487e00,
            9.23453652e00,
            2.15796783e-02,
            -8.81134487e01,
            -2.12131911e00,
            3.08859341e-01,
            9.90149054e01,
            -2.58589571e00,
            4.21646546e-01,
            3.91986770e01,
            2.93997264e00,
            4.68431323e-01,
        ]
    )
    u_eq = (
        p[0] * pybamm.exp(-p[1] * sto)
        + p[2]
        + p[3] * pybamm.tanh(p[4] * (sto - p[5]))
        + p[6] * pybamm.tanh(p[7] * (sto - p[8]))
        + p[9] * pybamm.tanh(p[10] * (sto - p[11]))
        + p[12] * pybamm.tanh(p[13] * (sto - p[14]))
        - 0.5 * R * T / F * pybamm.log(sto)
    )
    return u_eq


def ocp_average(sto):
    return (ocp_lithiation(sto) + ocp_delithiation(sto)) / 2


def exchange_current_density_lithiation(c_e, c_s_surf, c_s_max, T):
    m_ref = 9e-7  # (A/m2)(m3/mol)**1.5
    return m_ref * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def exchange_current_density_delithiation(c_e, c_s_surf, c_s_max, T):
    m_ref = 6e-7  # (A/m2)(m3/mol)**1.5
    return m_ref * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def exchange_current_density_average(sto):
    return (
        exchange_current_density_lithiation(sto)
        + exchange_current_density_delithiation(sto)
    ) / 2


parameter_values.update(
    {
        "Negative electrode OCP [V]": ocp_average,
        "Negative electrode lithiation OCP [V]": ocp_lithiation,
        "Negative electrode delithiation OCP [V]": ocp_delithiation,
        "Negative electrode exchange-current density [A.m-2]"
        "": exchange_current_density_average,
        "Negative electrode lithiation exchange-current density [A.m-2]"
        "": exchange_current_density_lithiation,
        "Negative electrode delithiation exchange-current density [A.m-2]"
        "": exchange_current_density_delithiation,
        "Negative particle diffusivity [m2.s-1]": 3.3e-14,
        "Negative particle lithiation diffusivity [m2.s-1]": 4e-14,
        "Negative particle delithiation diffusivity [m2.s-1]": 2.6e-14,
    },
    check_already_exists=False,
)

# Create experiment and run simulation
experiment = pybamm.Experiment(
    ["Discharge at 1 C until 2.5 V", "Charge at 1 C until 4.2 V"]
)
sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)
sim.solve()

# Plot
sim.plot(
    [
        "X-averaged negative particle surface concentration [mol.m-3]",
        "X-averaged positive particle surface concentration [mol.m-3]",
        "X-averaged negative electrode exchange current density [A.m-2]",
        "X-averaged positive electrode exchange current density [A.m-2]",
        "X-averaged negative particle concentration [mol.m-3]",
        "X-averaged positive particle concentration [mol.m-3]",
        "Current [A]",
        "Voltage [V]",
    ]
)
