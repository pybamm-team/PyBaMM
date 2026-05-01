import pytest

import pybamm


def _layer_domain(i):
    return f"cell layer {i}"


class TestMultiLayer3DThermalSPM:
    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="num_physical_layers"):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(num_physical_layers=1)
        with pytest.raises(ValueError, match="num_subdivisions"):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(
                num_physical_layers=4, num_subdivisions=1
            )
        with pytest.raises(ValueError, match="divisible"):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(
                num_physical_layers=5, num_subdivisions=2
            )
        with pytest.raises(ValueError, match="connection"):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(connection="weird")
        with pytest.raises((NotImplementedError, pybamm.OptionError)):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(
                options={"cell geometry": "cylindrical", "dimensionality": 3}
            )
        # Mixing legacy and new kwargs must raise.
        with pytest.raises(ValueError, match="legacy"):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(
                num_physical_layers=6, num_layers=3
            )

    @pytest.mark.parametrize("num_layers", [2, 3, 5])
    def test_parallel_equation_counts(self, num_layers):
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(
            num_layers=num_layers, connection="parallel"
        )
        # rhs: per-layer c_s_n, c_s_p, T  -> 3 * num_layers
        assert len(m.rhs) == 3 * num_layers
        # algebraic: per-layer T_av link + per-layer current fraction
        assert len(m.algebraic) == 2 * num_layers
        assert len(m.thermal_variables) == num_layers

    @pytest.mark.parametrize("num_layers", [2, 3, 5])
    def test_series_equation_counts(self, num_layers):
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(
            num_layers=num_layers, connection="series"
        )
        assert len(m.rhs) == 3 * num_layers
        # algebraic: only T_av links (no current fractions)
        assert len(m.algebraic) == num_layers

    def test_geometry_layer_domains(self):
        num_layers = 3
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(num_layers=num_layers)
        geom = m.default_geometry
        for i in range(num_layers):
            assert _layer_domain(i) in geom
            layer_geo = geom[_layer_domain(i)]
            assert set(layer_geo.keys()) == {"x", "y", "z"}
        # x-bounds must be contiguous across layers
        x0_max = geom[_layer_domain(0)]["x"]["max"]
        x1_min = geom[_layer_domain(1)]["x"]["min"]
        # Symbolically equal expressions
        assert str(x0_max) == str(x1_min)

    def test_submesh_and_spatial_methods(self):
        num_layers = 3
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(num_layers=num_layers)
        submeshes = m.default_submesh_types
        methods = m.default_spatial_methods
        for i in range(num_layers):
            name = _layer_domain(i)
            assert isinstance(submeshes[name], pybamm.ScikitFemGenerator3D)
            assert isinstance(methods[name], pybamm.ScikitFiniteElement3D)

    def test_thermal_interface_bcs(self):
        """All thermal faces use Neumann BCs (both cooling and contact)."""
        num_layers = 3
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(num_layers=num_layers)
        for i in range(num_layers):
            T_i = m.thermal_variables[i]
            bcs = m.boundary_conditions[T_i]
            assert set(bcs.keys()) == {
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "z_min",
                "z_max",
            }
            # All BCs Neumann (cooling on external faces, contact-resistance
            # flux on internal interfaces).
            for face, (_, bc_type) in bcs.items():
                assert bc_type == "Neumann", (i, face, bc_type)

    def test_invalid_contact_resistance(self):
        with pytest.raises(ValueError, match="thermal_contact_resistance"):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(thermal_contact_resistance=0.0)

    def test_invalid_mesh_h(self):
        with pytest.raises(ValueError, match="mesh_h"):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(mesh_h=0.0)
        with pytest.raises(ValueError, match="mesh_h"):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(mesh_h=-0.1)

    def test_mesh_h_threads_to_submesh(self):
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(num_layers=2, mesh_h=0.05)
        sm = m.default_submesh_types
        for i in range(2):
            gen = sm[f"cell layer {i}"]
            assert isinstance(gen, pybamm.ScikitFemGenerator3D)
            assert str(gen.gen_params.get("h")) == "0.05"

    def test_output_variables_present(self):
        num_layers = 3
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(num_layers=num_layers)
        for i in range(num_layers):
            assert f"Layer {i} temperature [K]" in m.variables
            assert f"Layer {i} average temperature [K]" in m.variables
            assert f"Layer {i} voltage [V]" in m.variables
            assert f"Layer {i} current [A]" in m.variables
            assert f"Layer {i} current fraction" in m.variables
        assert "Voltage [V]" in m.variables
        assert "Temperature spread [K]" in m.variables

    def test_build_and_short_discharge(self):
        """End-to-end discretization + short solve."""
        model = pybamm.lithium_ion.MultiLayer3DThermalSPM(num_layers=2)
        param = pybamm.ParameterValues("Marquis2019")
        # Inject model-specific defaults (R_th + stack capacity).
        model.apply_stack_scaling(param, verbose=False)
        for k in [
            "Total",
            "Left face",
            "Right face",
            "Front face",
            "Back face",
            "Bottom face",
            "Top face",
        ]:
            param.update({f"{k} heat transfer coefficient [W.m-2.K-1]": 10})
        var_pts = {
            "x_n": 10,
            "x_s": 10,
            "x_p": 10,
            "r_n": 15,
            "r_p": 15,
            "x": None,
            "y": None,
            "z": None,
        }
        exp = pybamm.Experiment(["Discharge at 1C for 60 seconds"])
        sim = pybamm.Simulation(
            model, parameter_values=param, var_pts=var_pts, experiment=exp
        )
        sol = sim.solve()
        # Symmetric 2-layer parallel: fractions should be ~0.5 and
        # temperatures essentially equal.
        f0 = float(sol["Layer 0 current fraction"].data[-1])
        f1 = float(sol["Layer 1 current fraction"].data[-1])
        assert abs(f0 - 0.5) < 1e-6
        assert abs(f1 - 0.5) < 1e-6
        spread = float(sol["Temperature spread [K]"].data[-1])
        assert spread < 1e-6

    # ------------------------------------------------------------------ #
    # layers_per_zone coarsening
    # ------------------------------------------------------------------ #
    def test_contact_resistance_is_a_parameter(self):
        """R_th should be a proper pybamm.Parameter overridable via PV."""
        key = pybamm.lithium_ion.MultiLayer3DThermalSPM.CONTACT_RESISTANCE_PARAM
        # default_parameter_values picks up the kwarg default.
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(
            num_layers=2, thermal_contact_resistance=5e-3
        )
        pv = m.default_parameter_values
        assert pv[key] == 5e-3
        # apply_stack_scaling injects the default into a fresh PV.
        pv2 = pybamm.ParameterValues("Marquis2019")
        assert key not in pv2.keys()
        m.apply_stack_scaling(pv2, verbose=False)
        assert pv2[key] == 5e-3
        # apply_stack_scaling does NOT clobber a user-set value.
        pv3 = pybamm.ParameterValues("Marquis2019")
        pv3.update({key: 7.0})
        m.apply_stack_scaling(pv3, verbose=False)
        assert pv3[key] == 7.0

    def test_invalid_layers_per_zone(self):
        # Legacy kwarg path: layers_per_zone=0 -> num_physical_layers=0.
        with pytest.raises(ValueError, match="num_physical_layers"):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(num_layers=3, layers_per_zone=0)
        with pytest.raises(ValueError, match="num_physical_layers"):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(num_layers=3, layers_per_zone=-2)

    def test_layers_per_zone_attributes(self):
        # Legacy kwargs still populate the attributes correctly.
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(num_layers=4, layers_per_zone=6)
        assert m.layers_per_zone == 6
        assert m.num_physical_layers == 24
        assert m.num_subdivisions == 4
        assert m.num_layers == 4  # alias for num_subdivisions

    def test_new_api_attributes(self):
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(
            num_physical_layers=24, num_subdivisions=4
        )
        assert m.num_physical_layers == 24
        assert m.num_subdivisions == 4
        assert m.num_layers == 4
        assert m.layers_per_zone == 6

    def test_new_api_default_subdivisions(self):
        # Default num_subdivisions = num_physical_layers (fully resolved).
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(num_physical_layers=5)
        assert m.num_subdivisions == 5
        assert m.layers_per_zone == 1

    def test_layers_per_zone_scales_geometry(self):
        num_layers = 3
        n = 5
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(
            num_layers=num_layers, layers_per_zone=n
        )
        geom = m.default_geometry
        # Each zone spans n*L_x; the last zone's x_max should equal the
        # first zone's x_max times num_layers (string-compare symbolic).
        x0 = geom[_layer_domain(0)]["x"]
        x_last = geom[_layer_domain(num_layers - 1)]["x"]
        # Symbolic evaluation against GeometricParameters
        import pybamm as _pb

        geo_p = _pb.GeometricParameters(m.options)
        # Materialise with the default parameter set
        pv = _pb.ParameterValues("Marquis2019")
        L_x_val = pv.process_symbol(geo_p.L_x).evaluate()
        x0_max_val = pv.process_symbol(x0["max"]).evaluate()
        x_last_max_val = pv.process_symbol(x_last["max"]).evaluate()
        import numpy as np

        assert np.isclose(x0_max_val, n * L_x_val)
        assert np.isclose(x_last_max_val, num_layers * n * L_x_val)

    def test_layers_per_zone_divides_current(self):
        """Per-unit-cell current should be the zone current divided by n."""
        n = 4
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(
            num_layers=2, connection="parallel", layers_per_zone=n
        )
        # The registered output "Layer i current [A]" should already be the
        # per-unit-cell current (zone / n).
        for i in range(2):
            assert f"Layer {i} per-unit-cell current [A]" in m.variables
            assert (
                m.variables[f"Layer {i} per-unit-cell current [A]"]
                is m.variables[f"Layer {i} current [A]"]
            )

    def test_apply_stack_scaling(self):
        m = pybamm.lithium_ion.MultiLayer3DThermalSPM(num_layers=5, layers_per_zone=4)
        pv = pybamm.ParameterValues("Marquis2019")
        Q0 = pv["Nominal cell capacity [A.h]"]
        m.apply_stack_scaling(pv, verbose=False)
        assert pv["Nominal cell capacity [A.h]"] == Q0 * 20

    def test_layers_per_zone_invariance_short_solve(self):
        """(num_layers=2, n=3) vs (num_layers=6, n=1) should give matching
        terminal voltage and stack average temperature for a short solve
        under symmetric cooling."""
        import numpy as np

        def build_and_solve(num_layers, n):
            model = pybamm.lithium_ion.MultiLayer3DThermalSPM(
                num_layers=num_layers,
                connection="parallel",
                layers_per_zone=n,
            )
            param = pybamm.ParameterValues("Marquis2019")
            # Symmetric cooling so both configurations behave uniformly.
            for k in [
                "Total",
                "Left face",
                "Right face",
                "Front face",
                "Back face",
                "Bottom face",
                "Top face",
            ]:
                param.update({f"{k} heat transfer coefficient [W.m-2.K-1]": 10})
            model.apply_stack_scaling(param, verbose=False)
            var_pts = {
                "x_n": 10,
                "x_s": 10,
                "x_p": 10,
                "r_n": 15,
                "r_p": 15,
                "x": None,
                "y": None,
                "z": None,
            }
            exp = pybamm.Experiment(["Discharge at 1C for 30 seconds"])
            sim = pybamm.Simulation(
                model, parameter_values=param, var_pts=var_pts, experiment=exp
            )
            return sim.solve()

        sol_a = build_and_solve(num_layers=2, n=3)
        sol_b = build_and_solve(num_layers=6, n=1)

        V_a = float(sol_a["Voltage [V]"].data[-1])
        V_b = float(sol_b["Voltage [V]"].data[-1])
        assert np.isclose(V_a, V_b, atol=5e-3)

        T_a = float(sol_a["Stack-averaged temperature [K]"].data[-1])
        T_b = float(sol_b["Stack-averaged temperature [K]"].data[-1])
        assert np.isclose(T_a, T_b, atol=0.2)
