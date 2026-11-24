"""Fit transient closure modes and run the transient dual-continuum model (DC1t).

1. Isolated sphere: fit the modes to the analytical closure variable, and
   check that DC1t then reproduces the Fickian DFN, including during a
   constant-voltage hold.
2. 3D electrode of Paten et al. (2026): fit the modes to the closure variable
   computed on the tomographic image, and compare with the DNS.
"""

import matplotlib.pyplot as plt
import numpy as np

import pybamm
import pybamm_model_zoo as zoo
from pybamm_model_zoo.dual_continuum import (
    fit_transient_closure,
    isolated_sphere_closure,
    parameter_sets,
)

DualContinuum = zoo.load("DualContinuum")
HALF_CELL = {"working electrode": "positive"}


def isolated_sphere():
    parameter_values = pybamm.lithium_ion.DFN(HALF_CELL).default_parameter_values
    radius = parameter_values["Positive particle radius [m]"]
    diffusivity = parameter_values["Positive particle diffusivity [m2.s-1]"]

    # Any s(t) can be fitted: here the analytical solution for a sphere
    t = np.geomspace(1e-2, 1e4, 400)
    fit = fit_transient_closure(
        t, isolated_sphere_closure(t, radius, diffusivity), diffusivity=diffusivity
    )
    print(fit, end="\n\n")

    model = DualContinuum(
        HALF_CELL,
        {
            "model type": "DC1t",
            "transient closure modes": fit.n_modes,
            "surface area": "spherical",
        },
    )
    parameter_values = model.default_parameter_values
    parameter_values.update(
        fit.parameter_values("Positive"), check_already_exists=False
    )
    experiment = pybamm.Experiment(
        [
            "Discharge at 2C for 15 minutes",
            "Hold at 3.75 V for 20 minutes",
            "Rest for 20 minutes",
        ]
    )
    solutions = {
        label: pybamm.Simulation(
            m, parameter_values=parameter_values, experiment=experiment, var_pts=pts
        ).solve()
        for label, m, pts in [
            ("DC1t", model, None),
            (
                "DFN",
                pybamm.lithium_ion.DFN(HALF_CELL),
                {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 20, "r_p": 80},
            ),
        ]
    }
    times = np.linspace(0, solutions["DFN"].t[-1], 500)
    dv = solutions["DC1t"]["Voltage [V]"](times) - solutions["DFN"]["Voltage [V]"](
        times
    )
    print(
        f"Isolated sphere, DC1t vs Fickian DFN: max |dV| = {1e3 * abs(dv).max():.2f} mV"
    )
    fig, (ax_voltage, ax_current) = plt.subplots(2, 1, sharex=True)
    # The curves overlap, so the DFN is drawn wide underneath DC1t
    for label, style in [("DFN", {"lw": 4, "alpha": 0.4}), ("DC1t", {"ls": "--"})]:
        solution = solutions[label]
        ax_voltage.plot(
            times / 60, solution["Voltage [V]"](times), label=label, **style
        )
        ax_current.plot(times / 60, solution["Current [A]"](times), **style)
    ax_voltage.set_ylabel("Voltage [V]")
    ax_current.set_ylabel("Current [A]")
    ax_current.set_xlabel("Time [min]")
    ax_voltage.set_title("Isolated sphere: DC1t and Fickian DFN, CC-CV-rest")
    ax_voltage.legend()
    fig.tight_layout()


def tomographic_electrode():
    parameter_values = pybamm.ParameterValues("Paten2026_3D_NMC")
    data = parameter_sets.load_3d_case_data()
    closure, dns = data["closure"], data["dns"]

    fit = fit_transient_closure(
        closure["Time [s]"],
        closure["Closure variable [mol.m-1.A-1]"],
        diffusivity=parameter_values["Positive particle diffusivity [m2.s-1]"],
    )
    print(fit, end="\n\n")

    current = parameter_values["Current function [A]"]
    experiment = pybamm.Experiment(
        [f"Discharge at {current} A until 3.0 V", "Rest for 900 seconds"]
    )
    models = {
        "DFN": pybamm.lithium_ion.DFN(HALF_CELL),
        "DC1": DualContinuum(HALF_CELL),
        "DC1t": DualContinuum(
            HALF_CELL, {"model type": "DC1t", "transient closure modes": fit.n_modes}
        ),
    }
    parameter_values.update(
        fit.parameter_values("Positive"), check_already_exists=False
    )

    fit.plot(show_plot=False)

    # Against the same time for every model, as in Paten et al. (2026): each
    # model stops at its own 3.0 V cut-off, so their rests start at different times
    t_dns, v_dns = dns["Time [s]"], dns["Voltage [V]"]
    end_dns = t_dns[np.nonzero(dns["Current density [A.m-2]"] == 0)[0][0] - 1]
    fig, (ax_full, ax_rest) = plt.subplots(1, 2, figsize=(10, 4))
    for ax in (ax_full, ax_rest):
        ax.plot(t_dns, v_dns, "k.", ms=2, label="DNS")
    for label, model in models.items():
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        times = np.linspace(0, solution.t[-1], 3000)
        for ax in (ax_full, ax_rest):
            ax.plot(times, solution["Voltage [V]"](times), label=label)
    ax_rest.set_xlim(end_dns - 15, end_dns + 210)
    ax_rest.set_ylim(2.95, 3.65)
    for ax in (ax_full, ax_rest):
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Voltage [V]")
    ax_full.set_title("Tomographic NMC electrode: discharge and rest")
    ax_rest.set_title("End of discharge and start of the rest")
    ax_full.legend()
    fig.tight_layout()


if __name__ == "__main__":
    isolated_sphere()
    tomographic_electrode()
    plt.show()
