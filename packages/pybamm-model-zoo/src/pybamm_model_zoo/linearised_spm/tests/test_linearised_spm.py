"""Physics tests for the reference model zoo entry.

These are the contributor's own tests (the zoo's "Layer B"): the contract suite
already checks that the model imports, is well posed, builds, and solves, so
these pin *physical* results instead.
"""

import numpy as np
import pytest

import pybamm
import pybamm_model_zoo as zoo

FARADAY = 96485.33212
PULSE = 60.0
#: Radial points needed to resolve the sqrt(D t) boundary layer of a short pulse.
PARTICLE_POINTS = 200


def half_cell(diffusivity, pulse=PULSE, particle_points=PARTICLE_POINTS):
    """A GITT pulse on the positive electrode against a lithium counter-electrode."""
    model = zoo.load("LinearisedSPM")({"working electrode": "positive"})
    parameter_values = model.default_parameter_values
    parameter_values["Positive particle diffusivity [m2.s-1]"] = diffusivity
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameter_values,
        var_pts={**model.default_var_pts, "r_p": particle_points},
    )
    return parameter_values, simulation.solve([0, pulse])


def recovered_diffusivity(parameter_values, solution, pulse=PULSE):
    """Invert the Weppner-Huggins relation for the diffusivity, as a GITT fit does."""
    times = np.linspace(pulse / 20, pulse, 60)
    voltage = solution["Voltage [V]"](times)
    gradient = solution["Positive electrode open-circuit potential gradient [V]"](
        times[0]
    )
    flux = (
        solution["X-averaged positive electrode interfacial current density [A.m-2]"](
            times[0]
        )
        / FARADAY
    )
    concentration_max = parameter_values[
        "Maximum concentration in positive electrode [mol.m-3]"
    ]
    slope = np.polyfit(np.sqrt(times), voltage, 1)[0]
    return (4 / np.pi) * (gradient * flux / (concentration_max * slope)) ** 2


class TestLinearisedSPM:
    def test_open_circuit_potential_is_the_tangent(self):
        model = zoo.load("LinearisedSPM")()
        solution = pybamm.Simulation(model).solve([0, 300])
        times = np.linspace(0, 300, 20)
        for domain in ("negative", "positive"):
            reported = solution[
                f"X-averaged {domain} electrode open-circuit potential [V]"
            ](times)
            capitalised = domain.capitalize()
            tangent = solution[
                f"{capitalised} electrode linearisation open-circuit potential [V]"
            ](times) + solution[
                f"{capitalised} electrode open-circuit potential gradient [V]"
            ](times) * (
                solution[f"X-averaged {domain} particle surface stoichiometry"](times)
                - solution[f"{capitalised} electrode linearisation stoichiometry"](
                    times
                )
            )
            np.testing.assert_allclose(reported, tangent, rtol=1e-12)

    def test_agrees_with_spm_at_the_linearisation_point(self):
        solution = pybamm.Simulation(zoo.load("LinearisedSPM")()).solve([0, 300])
        reference = pybamm.Simulation(pybamm.lithium_ion.SPM()).solve([0, 300])
        np.testing.assert_allclose(
            solution["Bulk open-circuit voltage [V]"](0.0),
            reference["Bulk open-circuit voltage [V]"](0.0),
            rtol=1e-12,
        )

    def test_gradient_is_the_derivative_of_the_parameter_sets_ocp(self):
        model = zoo.load("LinearisedSPM")()
        solution = pybamm.Simulation(model).solve([0, 300])
        parameter_values = model.default_parameter_values
        for domain in ("Negative", "Positive"):
            stoichiometry = solution[f"{domain} electrode linearisation stoichiometry"](
                0.0
            )
            step = 1e-6
            ocp = parameter_values[f"{domain} electrode OCP [V]"]
            expected = (ocp(stoichiometry + step) - ocp(stoichiometry - step)) / (
                2 * step
            )
            np.testing.assert_allclose(
                solution[f"{domain} electrode open-circuit potential gradient [V]"](
                    0.0
                ),
                expected,
                rtol=1e-5,
            )

    def test_gitt_pulse_recovers_the_diffusivity(self):
        diffusivity = 1e-14
        parameter_values, solution = half_cell(diffusivity, pulse=5.0)
        recovered = recovered_diffusivity(parameter_values, solution, pulse=5.0)
        np.testing.assert_allclose(recovered, diffusivity, rtol=0.05)

    def test_weppner_huggins_bias_shrinks_with_the_pulse(self):
        """The residual is the sqrt(t) approximation, so it must vanish with it."""
        diffusivity = 1e-14
        errors = []
        for pulse in (80.0, 20.0, 5.0):
            parameter_values, solution = half_cell(diffusivity, pulse=pulse)
            recovered = recovered_diffusivity(parameter_values, solution, pulse=pulse)
            errors.append(abs(recovered / diffusivity - 1))
        assert errors[0] > errors[1] > errors[2], errors

    def test_rejects_an_open_circuit_potential_it_would_ignore(self):
        with pytest.raises(pybamm.OptionError, match=r"open-circuit potential"):
            zoo.load("LinearisedSPM")({"open-circuit potential": "current sigmoid"})
