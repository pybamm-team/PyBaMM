"""Physics tests for the multilayer 3D thermal stack.

The contract suite already checks that the model imports, is well posed,
builds, and solves. These pin what the stack has to get right: each zone reduces
to PyBaMM's own model, energy and lithium are conserved, and cooling one face
redistributes the current the way the kinetics say it must.
"""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

import pybamm
from pybamm_model_zoo import _compat
from pybamm_model_zoo.multilayer_3d_thermal import (
    MultiLayer3DThermalDFN,
    MultiLayer3DThermalSPM,
    MultiLayer3DThermalSPMe,
)
from pybamm_model_zoo.multilayer_3d_thermal.model import FACES

MODELS = [MultiLayer3DThermalSPM, MultiLayer3DThermalSPMe, MultiLayer3DThermalDFN]
#: Each zone's electrochemistry, as PyBaMM implements it for a single cell.
REFERENCES = {
    MultiLayer3DThermalSPM: pybamm.lithium_ion.SPM,
    MultiLayer3DThermalSPMe: pybamm.lithium_ion.SPMe,
    MultiLayer3DThermalDFN: pybamm.lithium_ion.DFN,
}
BOX_FACES = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}


def cooled(model, h=10.0, **faces):
    """The model's defaults, every face cooled at ``h`` unless named in ``faces``."""
    parameter_values = model.default_parameter_values
    parameter_values.update(
        {
            f"{face} face heat transfer coefficient [W.m-2.K-1]": faces.get(face, h)
            for face in FACES
        }
    )
    return parameter_values


def solve(model, parameter_values, t_end=None, experiment=None):
    simulation = pybamm.Simulation(
        model, parameter_values=parameter_values, experiment=experiment
    )
    return simulation.solve() if experiment else simulation.solve([0, t_end])


class TestConstruction:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"num_physical_layers": 1}, "num_physical_layers"),
            ({"num_physical_layers": 4, "num_subdivisions": 1}, "num_subdivisions"),
            ({"num_physical_layers": 5, "num_subdivisions": 2}, "divisible"),
            ({"connection": "diagonal"}, "connection"),
            ({"mesh_h": 0.0}, "mesh_h"),
            ({"mesh_h": -0.1}, "mesh_h"),
            ({"options": {"cell geometry": "cylindrical"}}, "pouch"),
        ],
    )
    def test_rejects_an_invalid_stack(self, kwargs, match):
        with pytest.raises(pybamm.OptionError, match=match):
            MultiLayer3DThermalSPM(**kwargs)

    @pytest.mark.parametrize("connection", ["parallel", "series"])
    @pytest.mark.parametrize("num_physical_layers", [2, 3, 5])
    def test_one_temperature_field_per_zone(self, num_physical_layers, connection):
        model = MultiLayer3DThermalSPM(num_physical_layers, connection=connection)
        # Two particles and a temperature field per zone. Each zone's average
        # temperature is algebraic, and in parallel so is its current fraction.
        assert len(model.rhs) == 3 * num_physical_layers
        algebraic_per_zone = 2 if connection == "parallel" else 1
        assert len(model.algebraic) == algebraic_per_zone * num_physical_layers

    def test_zones_tile_the_stack(self):
        model = MultiLayer3DThermalSPM(num_physical_layers=6, num_subdivisions=3)
        parameter_values = model.default_parameter_values
        geometry = model.default_geometry
        L_x = parameter_values.evaluate(model.param.L_x)
        bounds = [
            parameter_values.evaluate(geometry[f"cell layer {i}"]["x"][end])
            for i in range(3)
            for end in ("min", "max")
        ]
        np.testing.assert_allclose(
            bounds, np.array([0, 2, 2, 4, 4, 6]) * L_x, rtol=1e-12, atol=0
        )

    def test_mesh_h_reaches_every_zone(self):
        model = MultiLayer3DThermalSPM(num_physical_layers=2, mesh_h=0.05)
        for i in range(2):
            generator = model.default_submesh_types[f"cell layer {i}"]
            assert isinstance(generator, pybamm.ScikitFemGenerator3D)
            assert generator.gen_params["h"] == 0.05
            assert isinstance(
                model.default_spatial_methods[f"cell layer {i}"],
                pybamm.ScikitFiniteElement3D,
            )

    def test_every_face_carries_a_heat_flux(self):
        model = MultiLayer3DThermalSPM(num_physical_layers=3)
        for temperature in model.thermal_variables:
            conditions = model.boundary_conditions[temperature]
            assert set(conditions) == BOX_FACES
            assert {kind for _, kind in conditions.values()} == {"Neumann"}

    def test_cites_each_zones_electrochemistry(self):
        _compat.reset_citations()
        MultiLayer3DThermalDFN(num_physical_layers=2)
        assert "Doyle1993" in _compat.cited_keys()


class TestParameters:
    def test_defaults_supply_the_model_parameters(self):
        model = MultiLayer3DThermalSPM(num_physical_layers=2)
        parameter_values = model.default_parameter_values
        assert (
            parameter_values[model.CONTACT_RESISTANCE_PARAM]
            == model.DEFAULT_CONTACT_RESISTANCE
        )
        for face in FACES:
            assert (
                parameter_values[f"{face} face heat transfer coefficient [W.m-2.K-1]"]
                == model.DEFAULT_FACE_HEAT_TRANSFER_COEFFICIENT
            )

    def test_stack_scaling_keeps_values_already_set(self):
        model = MultiLayer3DThermalSPM(num_physical_layers=2)
        parameter_values = pybamm.ParameterValues("Marquis2019")
        parameter_values.update(
            {
                model.CONTACT_RESISTANCE_PARAM: 7.0,
                "Left face heat transfer coefficient [W.m-2.K-1]": 50.0,
            },
            check_already_exists=False,
        )
        model.apply_stack_scaling(parameter_values)
        assert parameter_values[model.CONTACT_RESISTANCE_PARAM] == 7.0
        assert (
            parameter_values["Left face heat transfer coefficient [W.m-2.K-1]"] == 50.0
        )
        assert (
            parameter_values["Right face heat transfer coefficient [W.m-2.K-1]"]
            == model.DEFAULT_FACE_HEAT_TRANSFER_COEFFICIENT
        )

    @pytest.mark.parametrize(
        ("connection", "cells_in_parallel"), [("parallel", 4), ("series", 2)]
    )
    def test_a_c_rate_is_one_unit_cells(self, connection, cells_in_parallel):
        """After stack scaling, 1C drives every unit cell at its own 1C.

        In series every zone carries the stack current, so only the unit cells
        within one zone share it.
        """
        model = MultiLayer3DThermalSPM(4, 2, connection=connection)
        assert model.cells_in_parallel == cells_in_parallel
        parameter_values = model.apply_stack_scaling(cooled(model))
        capacity = pybamm.ParameterValues("Marquis2019")["Nominal cell capacity [A.h]"]
        solution = solve(
            model,
            parameter_values,
            experiment=pybamm.Experiment(["Discharge at 1C for 60 seconds"]),
        )
        times = np.linspace(0, 60, 5)
        for i in range(2):
            np.testing.assert_allclose(
                solution[f"Layer {i} per-unit-cell current [A]"](times),
                capacity,
                rtol=1e-6,
            )
        if connection == "series":
            np.testing.assert_allclose(
                solution["Voltage [V]"](times),
                solution["Layer 0 voltage [V]"](times)
                + solution["Layer 1 voltage [V]"](times),
                rtol=1e-12,
            )


class TestPhysics:
    @pytest.mark.parametrize("model_class", MODELS)
    def test_each_zone_reduces_to_pybamm_model_when_isothermal(self, model_class):
        """Held isothermal and symmetric, each zone is the reference model's cell.

        Heavy cooling pins the stack to ambient, and doubling the current gives
        each of the two zones the reference cell's own current.
        """
        model = model_class(num_physical_layers=2)
        parameter_values = cooled(model, h=1e4)
        parameter_values["Current function [A]"] = (
            2 * parameter_values["Current function [A]"]
        )
        solution = solve(model, parameter_values, 1800)
        reference = solve(
            REFERENCES[model_class](), pybamm.ParameterValues("Marquis2019"), 1800
        )
        times = np.linspace(0, 1800, 50)
        np.testing.assert_allclose(
            solution["Stack-averaged temperature [K]"](times), 298.15, atol=1e-3
        )
        np.testing.assert_allclose(
            solution["Voltage [V]"](times),
            reference["Voltage [V]"](times),
            rtol=0,
            atol=1e-3,
        )

    def test_an_insulated_stack_keeps_every_joule(self):
        """Insulated, the stack stores exactly the heat it generates.

        This is what pins the interface coupling: heat has to leave one zone
        exactly as it enters the next.
        """
        model = MultiLayer3DThermalSPM(num_physical_layers=3)
        parameter_values = cooled(model, h=0.0)
        parameter_values["Current function [A]"] = 4.0
        solution = solve(model, parameter_values, 600)
        times = np.linspace(0, 600, 400)
        heat_capacity = parameter_values.evaluate(
            model.param.rho_c_p_eff(model.param.T_init)
        )
        initial = parameter_values.evaluate(model.param.T_init)
        stored = heat_capacity * sum(
            solution[f"Layer {i} average temperature [K]"](times) - initial
            for i in range(3)
        )
        generated = sum(
            cumulative_trapezoid(
                solution[f"Layer {i} heat generation [W.m-3]"](times), times, initial=0
            )
            for i in range(3)
        )
        np.testing.assert_allclose(stored, generated, rtol=0, atol=1e-3 * generated[-1])

    @pytest.mark.parametrize(
        "model_class", [MultiLayer3DThermalSPM, MultiLayer3DThermalSPMe]
    )
    def test_every_volt_lost_is_dissipated(self, model_class):
        """Without entropic heat, a zone's heat is its current times its lost voltage.

        An overpotential or ohmic drop that lowered the voltage without heating
        the zone, or heated it without lowering the voltage, would break this.
        """
        model = model_class(num_physical_layers=2)
        parameter_values = cooled(model)
        parameter_values.update(
            {
                f"{electrode} electrode OCP entropic change [V.K-1]": 0
                for electrode in ("Negative", "Positive")
            }
        )
        solution = solve(model, parameter_values, 1800)
        times = np.linspace(0, 1800, 20)
        unit_cell_volume = parameter_values.evaluate(model.param.L_x * model.param.A_cc)
        for i in range(2):
            heat = solution[f"Layer {i} heat generation [W.m-3]"](times)
            lost = solution[f"Layer {i} per-unit-cell current [A]"](times) * (
                solution[f"Layer {i} surface open-circuit voltage [V]"](times)
                - solution[f"Layer {i} voltage [V]"](times)
            )
            np.testing.assert_allclose(heat * unit_cell_volume, lost, rtol=1e-8)

    @pytest.mark.parametrize(
        "model_class", [MultiLayer3DThermalSPMe, MultiLayer3DThermalDFN]
    )
    def test_the_electrolyte_conserves_lithium(self, model_class):
        model = model_class(num_physical_layers=2)
        solution = solve(model, cooled(model), 1800)
        times = np.linspace(0, 1800, 20)
        for i in range(2):
            lithium = solution[
                f"Layer {i} total lithium in electrolyte per unit cell [mol]"
            ](times)
            np.testing.assert_allclose(lithium, lithium[0], rtol=1e-9)

    @pytest.mark.parametrize("model_class", MODELS)
    def test_a_symmetric_stack_shares_current_evenly(self, model_class):
        model = model_class(num_physical_layers=2)
        solution = solve(model, cooled(model), 600)
        times = np.linspace(0, 600, 10)
        for i in range(2):
            np.testing.assert_allclose(
                solution[f"Layer {i} current fraction"](times), 0.5, rtol=1e-6
            )
        np.testing.assert_allclose(
            solution["Temperature spread [K]"](times), 0, atol=1e-6
        )

    def test_cooling_one_face_draws_current_to_the_warm_side(self):
        """Cooled from the left, the stack warms to the right, and part-way through
        a discharge the warmer zones carry more of the current, since their
        kinetics and diffusion are faster. Near the end of a full discharge their
        lower state of charge hands it back, so this stops at two thirds.
        """
        model = MultiLayer3DThermalSPM(num_physical_layers=4)
        parameter_values = model.apply_stack_scaling(cooled(model, h=0.1, Left=50.0))
        parameter_values[model.CONTACT_RESISTANCE_PARAM] = 1e-2
        solution = solve(
            model,
            parameter_values,
            experiment=pybamm.Experiment(["Discharge at 2C for 20 minutes"]),
        )
        end = solution.t[-1]
        temperatures = np.array(
            [solution[f"Layer {i} average temperature [K]"](end) for i in range(4)]
        )
        fractions = np.array(
            [solution[f"Layer {i} current fraction"](end) for i in range(4)]
        )
        assert np.all(np.diff(temperatures) > 0), temperatures
        assert np.all(np.diff(fractions) > 0), fractions
        np.testing.assert_allclose(fractions.sum(), 1, rtol=1e-10)
        np.testing.assert_allclose(
            solution["Temperature spread [K]"](end),
            temperatures[-1] - temperatures[0],
            rtol=1e-10,
        )

    def test_coarse_zones_reproduce_a_resolved_stack(self):
        """Lumping unit cells into fewer zones leaves a uniformly cooled stack alone."""

        def stack(num_subdivisions):
            model = MultiLayer3DThermalSPM(6, num_subdivisions)
            return solve(
                model,
                model.apply_stack_scaling(cooled(model)),
                experiment=pybamm.Experiment(["Discharge at 1C for 10 minutes"]),
            )

        coarse, resolved = stack(2), stack(6)
        times = np.linspace(0, 600, 30)
        np.testing.assert_allclose(
            coarse["Voltage [V]"](times), resolved["Voltage [V]"](times), atol=1e-5
        )
        np.testing.assert_allclose(
            coarse["Stack-averaged temperature [K]"](times),
            resolved["Stack-averaged temperature [K]"](times),
            atol=1e-3,
        )
