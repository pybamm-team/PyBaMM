import pytest

import pybamm


def _layer_domain(i):
    return f"cell layer {i}"


class TestMultiLayer3DThermalSPM:
    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="num_layers"):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(num_layers=1)
        with pytest.raises(ValueError, match="connection"):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(connection="weird")
        with pytest.raises(
            (NotImplementedError, pybamm.OptionError)
        ):
            pybamm.lithium_ion.MultiLayer3DThermalSPM(
                options={"cell geometry": "cylindrical", "dimensionality": 3}
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
            pybamm.lithium_ion.MultiLayer3DThermalSPM(
                thermal_contact_resistance=0.0
            )

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
        for k in [
            "Total",
            "Left face",
            "Right face",
            "Front face",
            "Back face",
            "Bottom face",
            "Top face",
        ]:
            param.update(
                {f"{k} heat transfer coefficient [W.m-2.K-1]": 10}
            )
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
