"""Physics tests for the DFN-based dual-continuum model.

The closures are pinned against exact limits: DC0 is PyBaMM's uniform-profile
particle, DC1 with the isolated-sphere closure variable -R/(5 D F) (the
analytical solution of the closure problem for a sphere) is PyBaMM's
quadratic-profile particle, and DC1t with the isolated sphere's transient
closure variable is PyBaMM's Fickian particle. The image-based surface area and
the lithium-foil option are pinned against reference voltages from the
standalone implementation, which Paten et al. (2026) validated against DNS
(tests/data/standalone_reference.csv).
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

import pybamm
import pybamm_model_zoo as zoo
from pybamm_model_zoo import _compat
from pybamm_model_zoo.dual_continuum import fit_transient_closure, parameter_sets

SOLVE_TIME = 3000
TIMES = np.linspace(0, SOLVE_TIME, 100)
HALF_CELL = {"working electrode": "positive"}
CELLS = [pytest.param({}, id="full cell"), pytest.param(HALF_CELL, id="half cell")]
#: DC options that reproduce the DFN's geometry and need no DC parameters.
DFN_LIKE = {"surface area": "spherical", "closure variable": "isolated sphere"}
AREA = "Positive electrode specific surface area from image"
REFERENCE = Path(__file__).parent / "data" / "standalone_reference.csv"
#: Changes to the half-cell defaults used for the reference voltages.
# these are arbitrary values made to test each option
REFERENCE_CHANGES = {
    f"{AREA} (AM-CBD) [m-1]": 5.0e4,
    f"{AREA} (AM-separator) [m-1]": 2.0e4,
    "CBD surface porosity": 0.3,
    "Separator surface porosity": 0.6,
}


def dual_continuum(options=None, dc_options=None):
    return zoo.load("DualContinuum")(options, dc_options)


def solve(model, parameter_values=None, solver=None):
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameter_values or model.default_parameter_values,
        solver=solver,
    )
    return simulation.solve([0, SOLVE_TIME], t_interp=TIMES)


def voltage(model, parameter_values=None, solver=None):
    return solve(model, parameter_values, solver)["Voltage [V]"](TIMES)


class TestDualContinuum:
    @pytest.mark.parametrize("options", CELLS)
    def test_dc0_is_uniform_profile(self, options):
        dfn = pybamm.lithium_ion.DFN({**options, "particle": "uniform profile"})
        np.testing.assert_allclose(
            voltage(
                dual_continuum(options, {**DFN_LIKE, "model type": "DC0"}),
                solver=tight_solver(),
            ),
            voltage(dfn, solver=tight_solver()),
            rtol=1e-7,
        )

    @pytest.mark.parametrize("options", CELLS)
    def test_isolated_sphere_closure_is_quadratic_profile(self, options):
        dfn = pybamm.lithium_ion.DFN({**options, "particle": "quadratic profile"})
        np.testing.assert_allclose(
            voltage(dual_continuum(options, DFN_LIKE), solver=tight_solver()),
            voltage(dfn, solver=tight_solver()),
            rtol=1e-7,
        )

    @pytest.mark.parametrize("sei", ["reaction limited", "ec reaction limited"])
    def test_sei_matches_dfn(self, sei):
        # PyBaMM's SEI options apply unchanged to the DC model
        dfn = pybamm.lithium_ion.DFN({"SEI": sei, "particle": "quadratic profile"})
        parameter_values = dfn.default_parameter_values
        solution = solve(
            dual_continuum({"SEI": sei}, DFN_LIKE), parameter_values, tight_solver()
        )
        np.testing.assert_allclose(
            solution["Voltage [V]"](TIMES),
            voltage(dfn, parameter_values, tight_solver()),
            rtol=1e-7,
        )
        thickness = solution["X-averaged negative SEI thickness [m]"](TIMES)
        assert thickness[-1] > thickness[0]

    @pytest.mark.parametrize("options", CELLS)
    def test_default_parameters_describe_the_dfn_cell(self, options):
        # Defaults are the DFN's geometry and the isolated-sphere closure at
        # the reference state (constant diffusivities here), with no foil scaling
        dfn = pybamm.lithium_ion.DFN({**options, "particle": "quadratic profile"})
        for dc_options in [{}, {"closure variable": "dimensionless"}]:
            np.testing.assert_allclose(
                voltage(dual_continuum(options, dc_options), solver=tight_solver()),
                voltage(dfn, solver=tight_solver()),
                rtol=1e-7,
            )

    def test_surface_area_from_image(self):
        model = dual_continuum(HALF_CELL)
        parameter_values = model.default_parameter_values
        parameter_values.update(
            {
                f"{AREA} (AM-electrolyte) [m-1]": 2.0e5,
                f"{AREA} (AM-CBD) [m-1]": 1.0e5,
                f"{AREA} (AM-separator) [m-1]": 4.0e4,
                "CBD surface porosity": 0.3,
                "Separator surface porosity": 0.4,
            }
        )
        solution = solve(model, parameter_values)
        expected = 2.0e5 + 0.3 * 1.0e5 + 0.4 * 4.0e4
        eps_s = parameter_values["Positive electrode active material volume fraction"]
        np.testing.assert_allclose(
            solution["Positive electrode surface area to volume ratio [m-1]"](
                t=TIMES[-1]
            ),
            expected,
        )
        np.testing.assert_allclose(
            solution["Positive electrode effective particle radius [m]"](t=TIMES[-1]),
            3 * eps_s / expected,
        )

    def test_lithium_foil_surface_porosity(self):
        model = dual_continuum(
            HALF_CELL,
            {
                "closure variable": "isolated sphere",
                "lithium foil surface porosity": "true",
            },
        )
        parameter_values = model.default_parameter_values
        porosity = parameter_values["Separator surface porosity"]
        reference = dual_continuum(HALF_CELL, DFN_LIKE)
        reference_values = reference.default_parameter_values
        j0 = reference_values[
            "Exchange-current density for lithium metal electrode [A.m-2]"
        ]
        reference_values[
            "Exchange-current density for lithium metal electrode [A.m-2]"
        ] = lambda c_e, c_Li, T: porosity * j0(c_e, c_Li, T)
        np.testing.assert_allclose(
            voltage(model, parameter_values),
            voltage(reference, reference_values),
            rtol=1e-7,
        )

    def test_reload_keeps_lithium_foil_surface_porosity(self):
        model = dual_continuum(HALF_CELL, {"lithium foil surface porosity": "true"})
        # pybamm.Serialise reassigns the options of a default-constructed model,
        # which would drop the DC options; that must raise
        data = pybamm.Serialise.serialise_custom_model(model)
        with pytest.raises(pybamm.ModelError, match="dc_options are not saved"):
            pybamm.Serialise.load_custom_model(data)
        with pytest.raises(pybamm.ModelError, match="dc_options are not saved"):
            pybamm.BaseModel.from_config(model.to_config())
        # Pickling and copying keep them
        for copy in [pickle.loads(pickle.dumps(model)), model.new_copy()]:
            assert copy.dc_options == model.dc_options
            assert type(copy.param) is type(model.param)

    @pytest.mark.parametrize("options", CELLS)
    def test_a_priori_close_to_implicit_at_low_rate(self, options):
        a_priori = dual_continuum(
            options, {"calculate surface concentration a priori": "true"}
        )
        np.testing.assert_allclose(
            voltage(a_priori), voltage(dual_continuum(options)), atol=5e-3
        )

    def test_lithium_conservation(self):
        solution = solve(dual_continuum())
        lithium = solution["Total lithium in particles [mol]"](TIMES)
        np.testing.assert_allclose(lithium, lithium[0], rtol=1e-10)

    def test_yang_uses_particle_radius(self):
        yang = dual_continuum(HALF_CELL, {"model type": "Yang"})
        dc1 = dual_continuum(HALF_CELL)
        parameter_values = yang.default_parameter_values
        larger = parameter_values.copy()
        larger["Positive particle radius [m]"] *= 2
        # With image-based areas the radius enters only Yang's closure
        assert (
            np.abs(voltage(yang, larger) - voltage(yang, parameter_values)).max() > 1e-3
        )
        np.testing.assert_allclose(
            voltage(dc1, larger), voltage(dc1, parameter_values), rtol=1e-9
        )

    @pytest.mark.parametrize(
        ("name", "function"),
        [
            ("Paten2026", parameter_sets.paten2026_dc1),
            ("Paten2026_DFN", parameter_sets.paten2026_dfn),
            ("Paten2026_3D_NMC", parameter_sets.paten2026_3d_case),
        ],
    )
    def test_registered_parameter_sets(self, name, function):
        values = function()
        assert isinstance(values, dict)
        assert values["chemistry"] == "lithium_ion"
        assert name in pybamm.parameter_sets
        _compat.reset_citations()
        loaded = pybamm.ParameterValues(name)
        assert "Paten2026" in _compat.cited_keys()
        assert set(loaded.keys()) == set(pybamm.ParameterValues(values).keys())
        assert "Paten et al." in pybamm.parameter_sets.get_docstring(name)

    def test_paten2026_parameter_sets(self):
        dc1 = pybamm.ParameterValues("Paten2026")
        assert dc1["Positive electrode s0 surface average"] == -500.38
        assert dc1[
            "Negative electrode specific surface area from image (AM-electrolyte) [m-1]"
        ] == pytest.approx(441818.3)
        for model_class, values in [
            (zoo.load("DualContinuum"), dc1),
            (pybamm.lithium_ion.DFN, pybamm.ParameterValues("Paten2026_DFN")),
        ]:
            # Bruggeman coefficients are the exact equivalent of the tortuosities
            np.testing.assert_allclose(
                voltage(model_class(), values),
                voltage(
                    model_class({"transport efficiency": "tortuosity factor"}), values
                ),
                rtol=1e-9,
            )

    def test_surface_area_enters_only_through_its_total(self):
        model = dual_continuum(HALF_CELL)
        split = model.default_parameter_values
        split.update(REFERENCE_CHANGES)
        total = split.copy()
        total.update(
            {
                f"{AREA} (AM-electrolyte) [m-1]": split[
                    f"{AREA} (AM-electrolyte) [m-1]"
                ]
                + 0.3 * 5.0e4
                + 0.6 * 2.0e4,
                f"{AREA} (AM-CBD) [m-1]": 0.0,
                f"{AREA} (AM-separator) [m-1]": 0.0,
            }
        )
        np.testing.assert_allclose(
            voltage(model, split), voltage(model, total), rtol=1e-9
        )

    @pytest.mark.parametrize(
        ("label", "dc_options"),
        [
            (
                "DC0",
                {
                    "model type": "DC0",
                    "calculate surface concentration a priori": "true",
                },
            ),
            ("DC1", {"model type": "DC1"}),
            (
                "DC1 a priori",
                {
                    "model type": "DC1",
                    "calculate surface concentration a priori": "true",
                },
            ),
        ],
    )
    def test_matches_standalone_reference(self, label, dc_options):
        # The standalone implementation scales the foil exchange current
        model = dual_continuum(
            HALF_CELL, {**dc_options, "lithium foil surface porosity": "true"}
        )
        parameter_values = model.default_parameter_values
        parameter_values.update(REFERENCE_CHANGES)
        reference = read_reference()
        new = voltage(model, parameter_values, tight_solver())
        np.testing.assert_allclose(TIMES, reference["Time [s]"])
        np.testing.assert_allclose(new, reference[f"{label} voltage [V]"], atol=1e-5)

    @pytest.mark.parametrize(
        ("model_type", "cited"),
        [("DC1", set()), ("DC1t", {"Paten2026b"}), ("Yang", {"Yang2025"})],
    )
    def test_citations(self, model_type, cited):
        _compat.reset_citations()
        dual_continuum(dc_options={"model type": model_type})
        keys = _compat.cited_keys() & {"Paten2026", "Paten2026b", "Yang2025"}
        assert keys == {"Paten2026"} | cited

    def test_options(self):
        with pytest.raises(pybamm.OptionError, match="not recognised"):
            dual_continuum(dc_options={"closure": "DC1"})
        with pytest.raises(
            pybamm.OptionError, match=r"Best matches are \['model type'\]"
        ):
            dual_continuum(dc_options={"model typ": "DC1"})
        with pytest.raises(
            pybamm.OptionError, match=r"Best matches are \['from image'\]"
        ):
            dual_continuum(dc_options={"surface area": "rom image"})
        with pytest.raises(pybamm.OptionError, match="Possible values are"):
            dual_continuum(dc_options={"model type": "xyz"})
        with pytest.raises(pybamm.OptionError, match="Invalid value"):
            dual_continuum(dc_options={"surface area": "components"})
        with pytest.raises(pybamm.OptionError, match="Invalid value"):
            dual_continuum(dc_options={"closure variable": "parameter"})
        with pytest.raises(pybamm.OptionError, match="a priori"):
            dual_continuum(
                dc_options={
                    "model type": "Yang",
                    "calculate surface concentration a priori": "true",
                }
            )
        with pytest.raises(pybamm.OptionError, match="requires galvanostatic"):
            dual_continuum(
                {"operating mode": "voltage"},
                {"calculate surface concentration a priori": "true"},
            )
        with pytest.raises(pybamm.OptionError, match="only applies to half cells"):
            dual_continuum(dc_options={"lithium foil surface porosity": "true"})
        with pytest.raises(pybamm.OptionError, match="single particle phase"):
            dual_continuum(
                COMPOSITE, {"calculate surface concentration a priori": "true"}
            )
        with pytest.raises(pybamm.OptionError, match="only applies to model type"):
            dual_continuum(dc_options={"transient closure modes": 4})
        for value in ["four", 0, True, 2.5]:
            with pytest.raises(pybamm.OptionError, match="transient closure modes"):
                dual_continuum(
                    dc_options={"model type": "DC1t", "transient closure modes": value}
                )
        assert (
            dual_continuum(
                dc_options={"model type": "DC1t", "transient closure modes": "3"}
            ).dc_options["transient closure modes"]
            == 3
        )


DC1T = {"model type": "DC1t"}
#: Fine radial mesh, so that the Fickian DFN is a near-exact reference.
FINE_DFN_MESH = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 160, "r_p": 160}
CC_CV_REST = pybamm.Experiment(
    [
        "Discharge at 2C for 15 minutes",
        "Hold at 3.75 V for 20 minutes",
        "Rest for 20 minutes",
    ]
)


class TestTransientDualContinuum:
    @pytest.mark.parametrize("options", CELLS)
    def test_defaults_match_fickian_dfn(self, options):
        # Default DC1t modes are fitted to the isolated sphere's s(t)
        dfn = pybamm.lithium_ion.DFN(options)
        reference = pybamm.Simulation(dfn, var_pts=FINE_DFN_MESH).solve(
            [0, SOLVE_TIME], t_interp=TIMES
        )
        np.testing.assert_allclose(
            voltage(dual_continuum(options, DC1T)),
            reference["Voltage [V]"](TIMES),
            atol=5e-4,
        )

    def test_constant_voltage_hold_matches_fickian_dfn(self):
        # The modes need no assumption on the current history
        model = dual_continuum(HALF_CELL, {**DC1T, **DFN_LIKE})
        solutions = [
            pybamm.Simulation(
                m,
                parameter_values=model.default_parameter_values,
                experiment=CC_CV_REST,
                var_pts=pts,
            ).solve()
            for m, pts in [
                (model, None),
                (pybamm.lithium_ion.DFN(HALF_CELL), FINE_DFN_MESH),
            ]
        ]
        times = np.linspace(0, solutions[1].t[-1], 400)
        for name, atol in [("Voltage [V]", 1e-3), ("Current [A]", 1e-4)]:
            np.testing.assert_allclose(
                solutions[0][name](times), solutions[1][name](times), atol=atol
            )

    @pytest.mark.parametrize("options", CELLS)
    def test_closure_variable_options_agree(self, options):
        # Defaults are the isolated sphere, in dimensional and dimensionless form
        expected = voltage(dual_continuum(options, DC1T))
        for closure in ["dimensionless", "isolated sphere"]:
            np.testing.assert_allclose(
                voltage(dual_continuum(options, {**DC1T, "closure variable": closure})),
                expected,
                rtol=1e-7,
            )

    def test_fast_modes_give_dc1(self):
        model = dual_continuum(HALF_CELL, {**DC1T, "transient closure modes": 4})
        parameter_values = model.default_parameter_values
        for m in range(1, 5):
            name = f"Positive electrode transient closure time constant {m} [s]"
            parameter_values[name] *= 1e-5
        # s(0) = 0, so the two differ only in the first ~1e-3 s
        np.testing.assert_allclose(
            voltage(model, parameter_values, tight_solver())[1:],
            voltage(dual_continuum(HALF_CELL), solver=tight_solver())[1:],
            rtol=1e-7,
        )

    def test_a_priori_is_the_galvanostatic_convolution(self):
        # Constant current: c_surf = c_vol + s(t) jbar, with s(t) the modes' sum
        model = dual_continuum(
            HALF_CELL,
            {
                **DC1T,
                "transient closure modes": 4,
                "calculate surface concentration a priori": "true",
            },
        )
        parameter_values = model.default_parameter_values
        prefix = "Positive electrode transient closure"
        s0 = parameter_values["Positive electrode s0 surface average"]
        s_of_t = s0 * sum(
            parameter_values[f"{prefix} weight {m}"]
            * (
                1
                - pybamm.exp(
                    -pybamm.t / parameter_values[f"{prefix} time constant {m} [s]"]
                )
            )
            for m in range(1, 5)
        )
        reference_values = parameter_values.copy()
        reference_values["Positive electrode s0 surface average"] = s_of_t
        reference = dual_continuum(
            HALF_CELL, {"calculate surface concentration a priori": "true"}
        )
        np.testing.assert_allclose(
            voltage(model, parameter_values, tight_solver()),
            voltage(reference, reference_values, tight_solver()),
            rtol=1e-7,
        )

    def test_lithium_conservation(self):
        solution = solve(dual_continuum(dc_options=DC1T))
        lithium = solution["Total lithium in particles [mol]"](TIMES)
        np.testing.assert_allclose(lithium, lithium[0], rtol=1e-10)

    def test_3d_case_against_dns(self):
        # Paten et al. (2026): closure data and DNS on a tomographic NMC electrode
        parameter_values = pybamm.ParameterValues("Paten2026_3D_NMC")
        data = parameter_sets.load_3d_case_data()
        closure, dns = data["closure"], data["dns"]
        fit = fit_transient_closure(
            closure["Time [s]"],
            closure["Closure variable [mol.m-1.A-1]"],
            diffusivity=4e-14,
        )
        parameter_values.update(fit.parameter_values(), check_already_exists=False)
        current = parameter_values["Current function [A]"]
        experiment = pybamm.Experiment(
            [f"Discharge at {current} A until 3.0 V", "Rest for 900 seconds"]
        )
        voltages = {}
        for label, dc_options in [
            ("DC1", {}),
            ("DC1t", {**DC1T, "transient closure modes": fit.n_modes}),
        ]:
            solution = pybamm.Simulation(
                dual_continuum(HALF_CELL, dc_options),
                parameter_values=parameter_values,
                experiment=experiment,
            ).solve()
            end = solution.cycles[0].steps[0].t[-1]
            discharge = dns["Time [s]"] < 1600
            voltages[label] = {
                "discharge": solution["Voltage [V]"](dns["Time [s]"][discharge])
                - dns["Voltage [V]"][discharge],
                "rest": solution["Voltage [V]"](end + np.array([10.0, 100.0, 900.0])),
                "end": end,
            }
        dns_end = 1741.9888
        dns_rest = np.interp(
            dns_end + np.array([10.0, 100.0, 900.0]),
            dns["Time [s]"],
            dns["Voltage [V]"],
        )
        dc1t = voltages["DC1t"]
        assert abs(dc1t["end"] - dns_end) < 5
        assert np.sqrt(np.mean(dc1t["discharge"] ** 2)) < 3e-3
        assert np.all(np.abs(dc1t["rest"] - dns_rest) < [0.08, 0.025, 0.005])
        # The steady closure misses the relaxation that the transient one captures
        assert abs(voltages["DC1"]["rest"][0] - dns_rest[0]) > 0.15


#: Graphite-silicon negative electrode; PyBaMM requires a surface form for it.
COMPOSITE = {"particle phases": ("2", "1"), "surface form": "algebraic"}
SINGLE_PHASE = {"surface form": "algebraic"}
EPS_N = "Negative electrode active material volume fraction"


def identical_phases():
    """Chen2020_composite's graphite as one phase, and as two identical halves."""
    base = pybamm.ParameterValues("Chen2020_composite")
    single, split = {}, {}
    for key, value in base.items():
        if not key.startswith(("Primary: ", "Secondary: ")):
            single[key] = split[key] = value
    # Phase-prefixed entries override the unprefixed leftovers of Chen2020
    for key, value in base.items():
        if key.startswith("Primary: "):
            name = key.removeprefix("Primary: ")
            single[name] = split[key] = split[f"Secondary: {name}"] = value
    eps = base[f"Primary: {EPS_N}"] + base[f"Secondary: {EPS_N}"]
    single[EPS_N] = eps
    split[f"Primary: {EPS_N}"] = split[f"Secondary: {EPS_N}"] = eps / 2
    return pybamm.ParameterValues(single), pybamm.ParameterValues(split)


class TestCompositeElectrode:
    @pytest.mark.parametrize("model_type", ["DC1", "DC1t", "Yang"])
    def test_identical_phases_give_one_phase(self, model_type):
        single, split = identical_phases()
        one, two = (
            dual_continuum(options, {"model type": model_type})
            for options in (SINGLE_PHASE, COMPOSITE)
        )
        np.testing.assert_allclose(
            voltage(two, two.add_dc_parameters(split), tight_solver()),
            voltage(one, one.add_dc_parameters(single), tight_solver()),
            rtol=1e-7,
        )

    def test_graphite_silicon_matches_fickian_dfn(self):
        # Each phase gets the isolated-sphere modes of its own R and D
        parameter_values = pybamm.ParameterValues("Chen2020_composite")
        model = dual_continuum(COMPOSITE, DC1T)
        dfn = pybamm.lithium_ion.DFN(COMPOSITE)
        var_pts = {
            name: 80 if name.startswith("r_") else points
            for name, points in dfn.default_var_pts.items()
        }
        reference = pybamm.Simulation(
            dfn, parameter_values=parameter_values, var_pts=var_pts
        ).solve([0, SOLVE_TIME], t_interp=TIMES)
        np.testing.assert_allclose(
            voltage(model, model.add_dc_parameters(parameter_values)),
            reference["Voltage [V]"](TIMES),
            atol=1e-3,
        )

    def test_parameter_names(self):
        parameter_values = pybamm.ParameterValues("Chen2020_composite")
        parameter_values.update(
            {"Positive electrode s0 surface average": -123.0},
            check_already_exists=False,
        )
        model = dual_continuum(COMPOSITE, DC1T)
        values = model.add_dc_parameters(parameter_values)
        # Prefixed only in the composite electrode; existing values are kept
        assert values["Positive electrode s0 surface average"] == -123.0
        assert "Negative electrode s0 surface average" not in values
        # Graphite (primary) and silicon (secondary): -R / (5 D F) for each
        for phase, radius, diffusivity in [
            ("Primary", 5.86e-6, 5.5e-14),
            ("Secondary", 1.52e-6, 1.67e-14),
        ]:
            np.testing.assert_allclose(
                values[f"{phase}: Negative electrode s0 surface average"],
                -radius / (5 * diffusivity * 96485.33212),
                rtol=1e-6,
            )
            assert (
                f"{phase}: Negative electrode transient closure time constant 8 [s]"
                in values
            )
        # PyBaMM's default set has no composite parameters: nothing is added
        defaults = model.default_parameter_values
        assert "Primary: Negative electrode s0 surface average" not in defaults


def tight_solver():
    """Tight tolerances, so that solver error does not mask model differences."""
    return pybamm.IDAKLUSolver(rtol=1e-7, atol=1e-8)


def read_reference():
    """Reference voltages by column name (the last comment line)."""
    with open(REFERENCE) as file:
        comments = [line for line in file if line.startswith("#")]
    names = comments[-1].lstrip("# ").strip().split(",")
    data = np.loadtxt(REFERENCE, delimiter=",")
    return dict(zip(names, data.T, strict=True))
