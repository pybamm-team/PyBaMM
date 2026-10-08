import numpy as np
import pytest

import pybamm


class TestEnergyBalance:
    @pytest.mark.parametrize(
        "options, parameter_set, current, initial_soc",
        [
            (
                {
                    "SEI": "solvent-diffusion limited",
                    "SEI on cracks": "true",
                    "particle mechanics": "swelling and cracking",
                    "lithium plating": "partially reversible",
                },
                "OKane2022",
                -10.0,
                0.1,
            ),
            (
                {
                    "particle phases": ("2", "1"),
                    "open-circuit potential": (("single", "current sigmoid"), "single"),
                    "SEI": "solvent-diffusion limited",
                    "lithium plating": "partially reversible",
                },
                "Bonkile2024",
                -10.0,
                0.1,
            ),
            (
                {"working electrode": "positive", "SEI": "reaction limited"},
                "Xu2019",
                2.4e-3,
                None,
            ),
        ],
    )
    def test_side_reaction_heat_matches_electrical_loss(
        self, options, parameter_set, current, initial_soc
    ):
        # Ohmic plus irreversible heat must equal the power drawn by all reactions
        # at their open-circuit potentials minus the terminal power
        model = pybamm.lithium_ion.DFN(
            {"calculate heat source for isothermal models": "true", **options}
        )
        param = pybamm.ParameterValues(parameter_set)
        # Xu2019 has no SEI growth, current collector or thermal parameters
        okane2022 = pybamm.ParameterValues("OKane2022")
        param.update(
            {k: okane2022[k] for k in okane2022 if k not in param},
            check_already_exists=False,
        )
        param.update(
            {
                "Current function [A]": current,
                "Ambient temperature [K]": 268.15,
                "Initial temperature [K]": 268.15,
                "Initial SEI thickness [m]": 50e-9,
            }
        )
        if parameter_set == "Xu2019":
            # Grow SEI fast enough on the lithium metal for its heat to matter
            param["SEI reaction exchange current density [A.m-2]"] = 1e-3
        sim = pybamm.Simulation(model, parameter_values=param)
        solution = sim.solve([0, 1800], initial_soc=initial_soc)
        t = solution.t

        area = param["Electrode width [m]"] * param["Electrode height [m]"]
        stored_power = 0
        for domain in ["negative", "positive"]:
            Domain = domain.capitalize()
            if model.options.electrode_types[domain] == "planar":
                # Lithium plating has zero open-circuit potential
                j_sei = solution[
                    "Negative electrode SEI interfacial current density [A.m-2]"
                ]
                stored_power += (
                    j_sei(t) * param["SEI open-circuit potential [V]"] * area
                )
                continue
            x = sim.mesh[f"{domain} electrode"].nodes
            phases = model.options.phases[domain]
            for phase in phases:
                phase_name = f"{phase} " if len(phases) > 1 else ""
                prefix = f"{phase.capitalize()}: " if len(phases) > 1 else ""
                U_sei = param[f"{prefix}SEI open-circuit potential [V]"]

                def field(name, prefix=f"{Domain} electrode {phase_name}", x=x):
                    return solution[f"{prefix}{name}"](t=t, x=x)

                power_density = field(
                    "volumetric interfacial current density [A.m-3]"
                ) * field("open-circuit potential [V]")
                for reaction in ["SEI ", "SEI on cracks "]:
                    power_density += (
                        field(
                            f"{reaction}volumetric interfacial current density [A.m-3]"
                        )
                        * U_sei
                    )
                # The default mesh is uniform within each electrode
                thickness = param[f"{Domain} electrode thickness [m]"]
                stored_power += power_density.mean(axis=0) * thickness * area

        heat = solution["Ohmic heating [W]"](t) + solution[
            "Irreversible electrochemical heating [W]"
        ](t)
        electrical_loss = -stored_power - solution["Current [A]"](t) * solution[
            "Voltage [V]"
        ](t)
        np.testing.assert_allclose(
            np.trapezoid(heat, t), np.trapezoid(electrical_loss, t), rtol=1e-3
        )
