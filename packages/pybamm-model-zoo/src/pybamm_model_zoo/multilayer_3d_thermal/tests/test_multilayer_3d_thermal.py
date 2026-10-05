"""Physics tests for the multilayer 3D thermal stack.

The contract suite already checks that the model imports, is well posed,
builds, and solves. These pin what the stack has to get right: each zone is
PyBaMM's own model under the options it is given, the stack holds the same
volume and heat as a lumped cell, energy and lithium are conserved, and cooling
one face redistributes the current the way the kinetics say it must.
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
BOX_FACES = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
#: Two particle phases in the negative electrode, the secondary with hysteresis.
COMPOSITE = {
    "particle phases": ("2", "1"),
    "open-circuit potential": (("single", "one-state hysteresis"), "single"),
}
#: The same, with Marcus-Hush-Chidsey kinetics on the negative electrode.
COMPOSITE_MHC = {
    **COMPOSITE,
    "intercalation kinetics": ("Marcus-Hush-Chidsey", "symmetric Butler-Volmer"),
}
#: Marcus-Hush-Chidsey suppresses the rate at a given j0 by tens of times at
#: lambda = 0.3 eV, so its j0 is scaled up to keep the discharge feasible.
MHC_J0_SCALE = 50.0
#: PyBaMM warns on building any one-state hysteresis submodel, whatever its inputs.
hysteresis_warning = pytest.mark.filterwarnings(
    "ignore:The definition of the hysteresis decay rate parameter:UserWarning"
)


def cooled(model, h=10.0, parameter_values=None, **faces):
    """The model's defaults, every face cooled at ``h`` unless named in ``faces``."""
    if parameter_values is None:
        parameter_values = model.default_parameter_values
    parameter_values.update(
        {
            f"{face} face heat transfer coefficient [W.m-2.K-1]": faces.get(face, h)
            for face in FACES
        },
        check_already_exists=False,
    )
    return parameter_values


def solve(model, parameter_values, t_end=None, experiment=None, **kwargs):
    simulation = pybamm.Simulation(
        model, parameter_values=parameter_values, experiment=experiment, **kwargs
    )
    return simulation.solve() if experiment else simulation.solve([0, t_end])


def scaled(function, factor):
    """``function`` times ``factor``, as a parameter function of the same inputs."""

    def scaled_function(*args):
        return factor * function(*args)

    return scaled_function


def composite_parameter_values(options=None):
    """``Chen2020_composite`` with what a lumped, hysteretic silicon phase needs,
    and, for Marcus-Hush-Chidsey kinetics, reorganization energies and a j0 on
    that rate law's scale."""
    parameter_values = pybamm.ParameterValues("Chen2020_composite")
    if options is not None and "intercalation kinetics" in options:
        for phase in ("Primary", "Secondary"):
            parameter_values[
                f"{phase}: Negative electrode reorganization energy [eV]"
            ] = 0.3
            name = f"{phase}: Negative electrode exchange-current density [A.m-2]"
            parameter_values[name] = scaled(parameter_values[name], MHC_J0_SCALE)
    lithiation = parameter_values["Secondary: Negative electrode lithiation OCP [V]"]
    delithiation = parameter_values[
        "Secondary: Negative electrode delithiation OCP [V]"
    ]
    parameter_values.update(
        {
            "Secondary: Negative particle lithiation hysteresis decay rate": 10.0,
            "Secondary: Negative particle delithiation hysteresis decay rate": 10.0,
            "Secondary: Initial hysteresis state in negative electrode": 0.0,
            "Secondary: Negative electrode OCP [V]": (
                lambda sto: (lithiation(sto) + delithiation(sto)) / 2
            ),
            "Negative electrode density [kg.m-3]": parameter_values[
                "Primary: Negative electrode density [kg.m-3]"
            ],
        },
        check_already_exists=False,
    )
    return parameter_values


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
            ({"options": {"thermal": "x-full"}}, "thermal"),
            ({"options": {"dimensionality": 1}}, "dimensionality"),
            ({"coating": "triple-sided"}, "coating"),
        ],
    )
    def test_rejects_an_invalid_stack(self, kwargs, match):
        with pytest.raises(pybamm.OptionError, match=match):
            MultiLayer3DThermalSPM(**kwargs)

    @pytest.mark.parametrize("connection", ["parallel", "series"])
    @pytest.mark.parametrize("num_physical_layers", [2, 3, 5])
    def test_one_zone_model_and_field_per_zone(self, num_physical_layers, connection):
        model = MultiLayer3DThermalSPM(num_physical_layers, connection=connection)
        zone = pybamm.lithium_ion.SPM({"thermal": "lumped", "cell geometry": "pouch"})
        # The lumped temperature becomes the field; its average and, in
        # parallel, the current fraction are algebraic.
        assert len(model.rhs) == len(zone.rhs) * num_physical_layers
        algebraic_per_zone = len(zone.algebraic) + (
            2 if connection == "parallel" else 1
        )
        assert len(model.algebraic) == algebraic_per_zone * num_physical_layers

    @pytest.mark.parametrize(
        ("coating", "foil_share"), [("double-sided", 0.5), ("single-sided", 1.0)]
    )
    def test_zones_tile_the_stack_of_whole_unit_cells(self, coating, foil_share):
        """A unit cell is its electrodes and separator plus its share of each foil:
        half of each, shared with its neighbours, when coated on both sides."""
        model = MultiLayer3DThermalSPM(
            num_physical_layers=6, num_subdivisions=3, coating=coating
        )
        parameter_values = model.default_parameter_values
        geometry = model.default_geometry
        L = parameter_values.evaluate(model.param.L_x) + foil_share * sum(
            parameter_values[f"{e} current collector thickness [m]"]
            for e in ("Negative", "Positive")
        )
        np.testing.assert_allclose(
            parameter_values.evaluate(model.unit_cell_thickness), L, rtol=1e-12
        )
        bounds = [
            parameter_values.evaluate(geometry[f"cell layer {i}"]["x"][end])
            for i in range(3)
            for end in ("min", "max")
        ]
        np.testing.assert_allclose(
            bounds, np.array([0, 2, 2, 4, 4, 6]) * L, rtol=1e-12, atol=0
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

    @hysteresis_warning
    def test_zones_take_the_options(self):
        model = MultiLayer3DThermalSPMe(2, options=COMPOSITE)
        assert model.zone_options["particle phases"] == ("2", "1")
        assert model.zone_options["thermal"] == "lumped"
        assert model.zone_options["surface temperature"] == "ambient"
        assert any("hysteresis state" in name for name in model.variables)

    def test_a_zone_model_builds_every_zone(self):
        built = []

        def zone_model(options):
            built.append(dict(options))
            return pybamm.lithium_ion.SPM(options)

        MultiLayer3DThermalSPMe(4, 2, zone_model=zone_model)
        assert len(built) == 2
        assert all(options["thermal"] == "lumped" for options in built)

    def test_a_zone_without_a_lumped_temperature_is_rejected(self):
        with pytest.raises(pybamm.ModelError, match="lumped temperature"):
            MultiLayer3DThermalSPM(
                2, zone_model=lambda options: pybamm.lithium_ion.SPM()
            )


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
        ("coating", "foil_share"), [("double-sided", 0.5), ("single-sided", 1.0)]
    )
    def test_zones_conduct_through_the_stack_in_series(self, coating, foil_share):
        model = MultiLayer3DThermalSPM(
            num_physical_layers=6, num_subdivisions=2, coating=coating
        )
        parameter_values = model.default_parameter_values
        T = model.param.T_init
        resistance = parameter_values.evaluate(model.zone_series_resistance(T))
        layers = [
            ("Negative current collector", "Negative current collector"),
            ("Negative electrode", "Negative electrode"),
            ("Separator", "Separator"),
            ("Positive electrode", "Positive electrode"),
            ("Positive current collector", "Positive current collector"),
        ]
        T_init = parameter_values["Initial temperature [K]"]
        per_cell = 0.0
        for thickness, conductivity in layers:
            k = parameter_values[f"{conductivity} thermal conductivity [W.m-1.K-1]"]
            k = k(T_init) if callable(k) else k
            share = foil_share if "collector" in thickness else 1.0
            per_cell += share * parameter_values[f"{thickness} thickness [m]"] / k
        np.testing.assert_allclose(resistance, 3 * per_cell, rtol=1e-12)
        # Series conduction is far below the in-plane mean the field carries.
        L = parameter_values.evaluate(model.unit_cell_thickness)
        in_plane = parameter_values.evaluate(
            model.per_unit_cell(model.param.lambda_eff(T))
        )
        assert L / per_cell < in_plane / 10

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
        np.testing.assert_allclose(
            solution["Discharge capacity [A.h]"](times),
            cells_in_parallel * capacity * times / 3600,
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
            model_class.ZONE_MODEL(), pybamm.ParameterValues("Marquis2019"), 1800
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

    @hysteresis_warning
    @pytest.mark.parametrize(
        "options", [COMPOSITE, COMPOSITE_MHC], ids=["hysteresis", "hysteresis-MHC"]
    )
    @pytest.mark.parametrize("model_class", MODELS)
    def test_composite_hysteretic_zones_reduce_to_pybamm_model(
        self, model_class, options
    ):
        """The options reach each zone: two phases, hysteresis, and a rate law,
        as PyBaMM's own model under the same options."""
        model = model_class(num_physical_layers=2, options=options)
        reference_values = composite_parameter_values(options)
        parameter_values = cooled(
            model,
            h=1e4,
            parameter_values=model.apply_stack_scaling(reference_values.copy()),
        )
        parameter_values["Current function [A]"] = (
            2 * reference_values["Current function [A]"]
        )
        solution = solve(model, parameter_values, 1200)
        reference = solve(model_class.ZONE_MODEL(options), reference_values, 1200)
        times = np.linspace(0, 1200, 25)
        np.testing.assert_allclose(
            solution["Voltage [V]"](times),
            reference["Voltage [V]"](times),
            rtol=0,
            atol=1e-3,
        )

    @pytest.mark.parametrize(
        ("coating", "foil_share"), [("double-sided", 0.5), ("single-sided", 1.0)]
    )
    @pytest.mark.parametrize("model_class", MODELS)
    def test_an_insulated_stack_heats_as_a_lumped_cell(
        self, model_class, coating, foil_share
    ):
        """Insulated and uniform, the stack heats exactly as PyBaMM's lumped cell
        whose foils are one unit cell's share of them.

        The same source over the same volume and heat capacity: a zone that
        spanned less than its unit cells would heat faster.
        """
        model = model_class(num_physical_layers=2, coating=coating)
        parameter_values = cooled(model, h=0.0)
        parameter_values["Current function [A]"] = (
            2 * parameter_values["Current function [A]"]
        )
        solution = solve(model, parameter_values, 600)
        reference_values = pybamm.ParameterValues("Marquis2019")
        reference_values["Total heat transfer coefficient [W.m-2.K-1]"] = 0.0
        for e in ("Negative", "Positive"):
            name = f"{e} current collector thickness [m]"
            reference_values[name] = foil_share * reference_values[name]
        reference = solve(
            model_class.ZONE_MODEL({"thermal": "lumped", "cell geometry": "pouch"}),
            reference_values,
            600,
        )
        rise = solution["Stack-averaged temperature [K]"](600) - 298.15
        reference_rise = reference["Volume-averaged cell temperature [K]"](600) - 298.15
        np.testing.assert_allclose(rise, reference_rise, rtol=1e-3)

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
        initial = parameter_values.evaluate(model.param.T_init)
        stored = sum(
            solution[f"Layer {i} heat capacity [J.K-1.m-3]"](times)
            * (solution[f"Layer {i} average temperature [K]"](times) - initial)
            for i in range(3)
        )
        generated = sum(
            cumulative_trapezoid(
                solution[f"Layer {i} heat generation [W.m-3]"](times), times, initial=0
            )
            for i in range(3)
        )
        np.testing.assert_allclose(stored, generated, rtol=0, atol=1e-3 * generated[-1])

    def test_the_stack_heat_budget_is_its_zones(self):
        model = MultiLayer3DThermalSPMe(num_physical_layers=4, num_subdivisions=2)
        solution = solve(model, cooled(model), 600)
        times = np.linspace(0, 600, 10)
        per_zone = sum(
            solution[f"Layer {i} Total heating [W]"](times) for i in range(2)
        )
        np.testing.assert_allclose(
            solution["Total heating [W]"](times), 2 * per_zone, rtol=1e-12
        )
        V_unit = model.default_parameter_values.evaluate(
            model.unit_cell_thickness * model.param.A_cc
        )
        np.testing.assert_allclose(
            solution["Layer 0 Total heating [W]"](times),
            solution["Layer 0 heat generation [W.m-3]"](times) * V_unit,
            rtol=1e-8,
        )

    @pytest.mark.parametrize(
        "model_class", [MultiLayer3DThermalSPMe, MultiLayer3DThermalDFN]
    )
    def test_the_electrolyte_conserves_lithium(self, model_class):
        model = model_class(num_physical_layers=2)
        solution = solve(model, cooled(model), 1800)
        times = np.linspace(0, 1800, 20)
        for i in range(2):
            lithium = solution[f"Layer {i} Total lithium in electrolyte [mol]"](times)
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

    @pytest.mark.parametrize("connection", ["parallel", "series"])
    def test_an_experiment_sets_the_voltage_cut_off(self, connection):
        """An experiment discharges to its own cut-off, below the parameter set's
        3.105 V: no zone carries a voltage limit of its own."""
        model = MultiLayer3DThermalSPM(2, connection=connection)
        parameter_values = model.apply_stack_scaling(cooled(model))
        limit = 3.0 if connection == "parallel" else 6.0
        solution = solve(
            model,
            parameter_values,
            experiment=pybamm.Experiment([f"Discharge at 1C until {limit} V"]),
        )
        np.testing.assert_allclose(
            solution["Voltage [V]"].entries[-1], limit, atol=1e-3
        )
        assert not any(
            "Layer" in event.name and "voltage" in event.name for event in model.events
        )

    def test_a_parallel_stack_starts_from_rest(self):
        """A uniform stack at rest carries no current in any zone, then shares a load."""
        model = MultiLayer3DThermalSPMe(num_physical_layers=4)
        parameter_values = model.apply_stack_scaling(cooled(model))
        capacity = parameter_values["Nominal cell capacity [A.h]"]
        parameter_values["Current function [A]"] = pybamm.Interpolant(
            np.array([0, 20, 20.5, 300]),
            np.array([0, 0, capacity, capacity]),
            pybamm.t,
        )
        solution = solve(model, parameter_values, 300)
        assert solution.t[-1] == pytest.approx(300)
        at_rest = np.array([0.0, 10.0])
        for i in range(4):
            np.testing.assert_allclose(
                solution[f"Layer {i} current [A]"](at_rest), 0, atol=1e-12
            )
        # Under load the cooled outer zones carry a little less than the core.
        loaded = [solution[f"Layer {i} current [A]"](200.0) for i in range(4)]
        np.testing.assert_allclose(sum(loaded), capacity, rtol=1e-9)
        np.testing.assert_allclose(loaded, capacity / 4, rtol=1e-3)

    def test_zones_that_differ_exchange_current_at_rest(self):
        """After a discharge cooled from the left the zones differ, and at rest they
        balance through the parallel connection: the cold zone, which carried
        less of the discharge, gives current to the warm one, which carried more.
        """
        model = MultiLayer3DThermalSPM(num_physical_layers=4)
        parameter_values = model.apply_stack_scaling(cooled(model, h=0.1, Left=50.0))
        parameter_values[model.CONTACT_RESISTANCE_PARAM] = 1e-2
        solution = solve(
            model,
            parameter_values,
            experiment=pybamm.Experiment(
                ["Discharge at 2C for 15 minutes", "Rest for 20 minutes"]
            ),
        )
        rest = solution.cycles[1]
        early, late = rest.t[0] + 1, rest.t[-1]
        currents = np.array(
            [
                [rest[f"Layer {i} current [A]"](t) for i in range(4)]
                for t in (early, late)
            ]
        )
        np.testing.assert_allclose(currents.sum(axis=1), 0, atol=1e-10)
        assert np.all(currents[:, 0] > 1e-4), currents
        assert np.all(currents[:, -1] < -1e-4), currents

    @pytest.mark.parametrize("num_subdivisions", [3, 4])
    def test_faces_surface_and_core_under_symmetric_cooling(self, num_subdivisions):
        """Cooled alike on both big faces, the faces agree and the core is hottest."""
        model = MultiLayer3DThermalSPM(num_subdivisions)
        parameter_values = cooled(model, h=0.1, Left=50.0, Right=50.0)
        parameter_values[model.CONTACT_RESISTANCE_PARAM] = 1e-2
        solution = solve(model, parameter_values, 600)
        end = {
            name: solution[name](600)
            for name in solution.all_models[0].variables
            if name.endswith("temperature [K]") and not name.startswith("Layer")
        }
        np.testing.assert_allclose(
            end["Left face temperature [K]"],
            end["Right face temperature [K]"],
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            end["Surface temperature [K]"], end["Left face temperature [K]"], rtol=1e-9
        )
        assert end["Surface temperature [K]"] < end["Stack-averaged temperature [K]"]
        assert end["Stack-averaged temperature [K]"] < end["Core temperature [K]"]
        np.testing.assert_allclose(
            solution["Core-to-skin temperature difference [K]"](600),
            end["Core temperature [K]"] - end["Surface temperature [K]"],
            rtol=1e-12,
        )

    def test_one_sided_cooling_names_the_cold_face(self):
        model = MultiLayer3DThermalSPM(4)
        parameter_values = cooled(model, h=0.1, Left=50.0)
        parameter_values[model.CONTACT_RESISTANCE_PARAM] = 1e-2
        solution = solve(model, parameter_values, 600)
        left = solution["Left face temperature [K]"](600)
        right = solution["Right face temperature [K]"](600)
        assert left < right
        np.testing.assert_allclose(
            solution["Surface temperature [K]"](600), (left + right) / 2, rtol=1e-12
        )

    @pytest.mark.parametrize(
        "model_class", [MultiLayer3DThermalSPMe, MultiLayer3DThermalDFN]
    )
    def test_heat_of_mixing_reaches_every_zone(self, model_class):
        """PyBaMM's own heat of mixing, on a single-phase cell where it builds:
        insulated, the stack heats as a lumped cell with the same option, and the
        mixing term appears in every zone's heat budget."""
        options = {"heat of mixing": "true"}
        model = model_class(
            num_physical_layers=2, options=options, coating="single-sided"
        )
        parameter_values = cooled(model, h=0.0)
        parameter_values["Current function [A]"] = (
            2 * parameter_values["Current function [A]"]
        )
        solution = solve(model, parameter_values, 600)
        reference_values = pybamm.ParameterValues("Marquis2019")
        reference_values["Total heat transfer coefficient [W.m-2.K-1]"] = 0.0
        reference = solve(
            model_class.ZONE_MODEL(
                {"thermal": "lumped", "cell geometry": "pouch", **options}
            ),
            reference_values,
            600,
        )
        np.testing.assert_allclose(
            solution["Stack-averaged temperature [K]"](600) - 298.15,
            reference["Volume-averaged cell temperature [K]"](600) - 298.15,
            rtol=1e-3,
        )
        for i in range(2):
            assert (
                np.abs(
                    solution[f"Layer {i} Heat of mixing [W]"](np.linspace(60, 600, 5))
                ).max()
                > 0
            )

    def test_lumped_thermal_capacity_is_per_unit_cell_volume(self):
        """With "use lumped thermal capacity", each zone carries "Cell heat capacity
        [J.K-1.m-3]" over its unit cells' volume, as a lumped cell does over its own."""
        options = {"use lumped thermal capacity": "true"}
        model = MultiLayer3DThermalSPM(2, options=options, coating="single-sided")
        parameter_values = cooled(model, h=0.0)
        parameter_values.update(
            {"Cell heat capacity [J.K-1.m-3]": 2.5e6}, check_already_exists=False
        )
        parameter_values["Current function [A]"] = (
            2 * parameter_values["Current function [A]"]
        )
        solution = solve(model, parameter_values, 600)
        np.testing.assert_allclose(
            solution["Layer 0 heat capacity [J.K-1.m-3]"](300), 2.5e6, rtol=1e-12
        )
        reference_values = pybamm.ParameterValues("Marquis2019")
        reference_values.update(
            {
                "Total heat transfer coefficient [W.m-2.K-1]": 0.0,
                "Cell heat capacity [J.K-1.m-3]": 2.5e6,
            },
            check_already_exists=False,
        )
        reference = solve(
            pybamm.lithium_ion.SPM(
                {"thermal": "lumped", "cell geometry": "pouch", **options}
            ),
            reference_values,
            600,
        )
        np.testing.assert_allclose(
            solution["Stack-averaged temperature [K]"](600) - 298.15,
            reference["Volume-averaged cell temperature [K]"](600) - 298.15,
            rtol=1e-3,
        )

    def test_a_lumped_surface_option_is_kept_on_the_stack_only(self):
        """ "surface temperature": "lumped" is recorded on the stack, where tools
        that read it find it, but the zones keep no casing of their own."""
        model = MultiLayer3DThermalSPMe(2, options={"surface temperature": "lumped"})
        assert model.options["surface temperature"] == "lumped"
        assert model.zone_options["surface temperature"] == "ambient"
        solution = solve(model, cooled(model, h=50.0), 600)
        np.testing.assert_allclose(
            solution["Surface temperature [K]"](600),
            (
                solution["Left face temperature [K]"](600)
                + solution["Right face temperature [K]"](600)
            )
            / 2,
            rtol=1e-12,
        )

    def test_a_zone_model_can_replace_a_submodel(self):
        """The hook's purpose: a zone built with build=False and a submodel swapped
        in before it is built. Swapping in the same lumped thermal submodel leaves
        the stack's solution unchanged."""

        def zone_model(options):
            zone = pybamm.lithium_ion.SPMe(options, build=False)
            zone.submodels["thermal"] = pybamm.thermal.Lumped(zone.param, zone.options)
            zone.build_model()
            return zone

        default = MultiLayer3DThermalSPMe(2)
        swapped = MultiLayer3DThermalSPMe(2, zone_model=zone_model)
        times = np.linspace(0, 600, 10)
        a = solve(default, cooled(default), 600)
        b = solve(swapped, cooled(swapped), 600)
        for name in ("Voltage [V]", "Stack-averaged temperature [K]"):
            np.testing.assert_allclose(b[name](times), a[name](times), rtol=1e-9)

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

    def test_the_through_stack_gradient_does_not_depend_on_zoning(self):
        """Cooled through both big faces, core-to-skin is the same in 2 zones as in
        12, and close to a uniformly heated slab conducting in series."""

        def core_to_skin(num_subdivisions):
            model = MultiLayer3DThermalSPM(12, num_subdivisions)
            parameter_values = model.apply_stack_scaling(
                cooled(model, h=0.0, Left=50.0, Right=50.0)
            )
            solution = solve(
                model,
                parameter_values,
                experiment=pybamm.Experiment(["Discharge at 1C for 20 minutes"]),
            )
            end = solution.t[-1]
            return model, parameter_values, solution, end

        coarse = core_to_skin(2)
        resolved = core_to_skin(12)
        differences = [
            s["Core-to-skin temperature difference [K]"](end)
            for _, _, s, end in (coarse, resolved)
        ]
        np.testing.assert_allclose(differences[0], differences[1], rtol=1e-2)
        model, parameter_values, solution, end = resolved
        heat = solution["Volume-averaged total heating [W.m-3]"](end)
        T = model.param.T_init
        per_cell = parameter_values.evaluate(model.zone_series_resistance(T))
        thickness = 12 * parameter_values.evaluate(model.unit_cell_thickness)
        slab = heat * thickness * per_cell * 12 / 8
        np.testing.assert_allclose(differences[1], slab, rtol=0.05)

    def test_pybamms_contact_resistance_applies_per_unit_cell(self):
        """The zones' own "contact resistance" option drops each unit cell's
        voltage by its own current times "Contact resistance [Ohm]"."""
        resistance = 0.01

        def stack(options):
            model = MultiLayer3DThermalSPM(2, options=options)
            parameter_values = cooled(model)
            parameter_values["Contact resistance [Ohm]"] = resistance
            return solve(model, parameter_values, 600)

        ideal = stack({})
        contact = stack({"contact resistance": "true"})
        times = np.linspace(0, 600, 10)
        unit_cell_current = ideal["Layer 0 per-unit-cell current [A]"](times)
        np.testing.assert_allclose(
            contact["Voltage [V]"](times),
            ideal["Voltage [V]"](times) - unit_cell_current * resistance,
            rtol=1e-7,
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
