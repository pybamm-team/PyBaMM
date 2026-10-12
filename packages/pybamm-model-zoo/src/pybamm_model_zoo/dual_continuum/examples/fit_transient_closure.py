"""Fit transient closure modes and run the transient dual-continuum model (DC1t).

1. Isolated sphere: fit the modes to the analytical closure variable, and
   check that DC1t then reproduces the Fickian DFN, including during a
   constant-voltage hold.
2. 3D electrode of Paten et al. (2026): fit the modes to the closure variable
   computed on the tomographic image, and compare with the DNS.
"""

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
    di = solutions["DC1t"]["Current [A]"](times) - solutions["DFN"]["Current [A]"](
        times
    )
    print(
        "Isolated sphere, DC1t vs Fickian DFN (CC, CV hold, rest): "
        f"max |dV| = {1e3 * np.abs(dv).max():.2f} mV, "
        f"max |dI| = {1e3 * np.abs(di).max():.3f} mA\n"
    )


def tomographic_electrode():
    parameter_values = parameter_sets.paten2026_3d_case()
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

    t_dns, v_dns = dns["Time [s]"], dns["Voltage [V]"]
    end_dns = t_dns[np.nonzero(dns["Current density [A.m-2]"] == 0)[0][0] - 1]
    print("3D electrode vs DNS: voltage [V] after the end of discharge")
    print("        end of discharge [s]   +1 s     +10 s    +100 s   +900 s")
    after = np.array([1, 10, 100, 900])
    print(
        f"DNS     {end_dns:20.0f}   "
        + "   ".join(f"{v:.4f}" for v in np.interp(end_dns + after, t_dns, v_dns))
    )
    for label, model in models.items():
        solution = pybamm.Simulation(
            model, parameter_values=parameter_values, experiment=experiment
        ).solve()
        end = solution.cycles[0].steps[0].t[-1]
        voltage = solution["Voltage [V]"](end + after)
        print(f"{label:6s}  {end:20.0f}   " + "   ".join(f"{v:.4f}" for v in voltage))


if __name__ == "__main__":
    isolated_sphere()
    tomographic_electrode()
