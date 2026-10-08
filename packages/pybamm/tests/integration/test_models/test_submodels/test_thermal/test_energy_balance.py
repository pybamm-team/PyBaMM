import numpy as np

import pybamm


class TestEnergyBalance:
    def test_side_reaction_heat_matches_electrical_loss(self):
        # Ohmic plus irreversible heat must equal the power drawn by all reactions
        # at their open-circuit potentials minus the terminal power
        model = pybamm.lithium_ion.DFN(
            {
                "calculate heat source for isothermal models": "true",
                "SEI": "solvent-diffusion limited",
                "SEI on cracks": "true",
                "particle mechanics": "swelling and cracking",
                "lithium plating": "partially reversible",
            }
        )
        param = pybamm.ParameterValues("OKane2022")
        param.update(
            {
                "Current function [A]": -10.0,
                "Ambient temperature [K]": 268.15,
                "Initial temperature [K]": 268.15,
                "Initial SEI thickness [m]": 50e-9,
            }
        )
        sim = pybamm.Simulation(model, parameter_values=param)
        solution = sim.solve([0, 1800], initial_soc=0.1)
        t = solution.t

        area = param["Electrode width [m]"] * param["Electrode height [m]"]
        U_sei = param["SEI open-circuit potential [V]"]
        stored_power = 0
        for domain in ["Negative", "Positive"]:
            x = sim.mesh[f"{domain.lower()} electrode"].nodes

            def field(name, domain=domain, x=x):
                return solution[f"{domain} electrode {name}"](t=t, x=x)

            power_density = field(
                "volumetric interfacial current density [A.m-3]"
            ) * field("open-circuit potential [V]")
            for reaction in ["SEI ", "SEI on cracks "]:
                power_density += (
                    field(f"{reaction}volumetric interfacial current density [A.m-3]")
                    * U_sei
                )
            # The default mesh is uniform within each electrode
            thickness = param[f"{domain} electrode thickness [m]"]
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
