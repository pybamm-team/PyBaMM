#
# Tests for the base model class
#
import pytest

import pybamm


class TestBatteryGeometry:
    @pytest.fixture(params=[0, 1, 2])
    def geometry(self, request):
        geometry = pybamm.battery_geometry(
            options={
                "particle size": "distribution",
                "dimensionality": request.param,
            },
        )
        return geometry

    def test_geometry_keys(self, geometry):
        for domain_geoms in geometry.values():
            assert all(
                isinstance(spatial_var, str)
                for spatial_var in domain_geoms.keys()
            )
        geometry.print_parameter_info()

    def test_geometry(self):
        geo = pybamm.geometric_parameters
        for cc_dimension in [0, 1, 2]:
            geometry = pybamm.battery_geometry(
                options={
                    "particle size": "distribution",
                    "dimensionality": cc_dimension,
                },
            )
            assert isinstance(geometry, pybamm.Geometry)
            assert "negative electrode" in geometry
            assert "negative particle" in geometry
            assert "negative particle size" in geometry
            assert geometry["negative electrode"]["x_n"]["min"] == 0
            assert geometry["negative electrode"]["x_n"]["max"] == geo.n.L
            if cc_dimension == 1:
                assert "tabs" in geometry["current collector"]

        geometry = pybamm.battery_geometry(include_particles=False)
        assert "negative particle" not in geometry

        geometry = pybamm.battery_geometry()
        assert "negative particle size" not in geometry

        geometry = pybamm.battery_geometry(form_factor="cylindrical")
        assert geometry["current collector"]["r_macro"]["position"] == 1

        geometry = pybamm.battery_geometry(
            form_factor="cylindrical", options={"dimensionality": 1}
        )
        assert geometry["current collector"]["r_macro"]["min"] == geo.r_inner
        assert geometry["current collector"]["r_macro"]["max"] == 1

        options = {"particle phases": "2"}
        geometry = pybamm.battery_geometry(options=options)
        geo = pybamm.GeometricParameters(options=options)
        assert geometry["negative primary particle"]["r_n_prim"]["min"] == 0
        assert geometry["negative primary particle"]["r_n_prim"]["max"] == geo.n.prim.R_typ
        assert geometry["negative secondary particle"]["r_n_sec"]["min"] == 0
        assert geometry["negative secondary particle"]["r_n_sec"]["max"] == geo.n.sec.R_typ
        assert geometry["positive primary particle"]["r_p_prim"]["min"] == 0
        assert geometry["positive primary particle"]["r_p_prim"]["max"] == geo.p.prim.R_typ
        assert geometry["positive secondary particle"]["r_p_sec"]["min"] == 0
        assert geometry["positive secondary particle"]["r_p_sec"]["max"] == geo.p.sec.R_typ

    def test_geometry_error(self):
        with pytest.raises(pybamm.GeometryError, match="Invalid current"):
            pybamm.battery_geometry(
                form_factor="cylindrical", options={"dimensionality": 2}
            )
        with pytest.raises(pybamm.GeometryError, match="Invalid form"):
            pybamm.battery_geometry(form_factor="triangle")


class TestReadParameters:
    # This is the most complicated geometry and should test the parameters are
    # all returned for the deepest dict
    def test_read_parameters(self):
        geo = pybamm.geometric_parameters
        L_n = geo.n.L
        L_s = geo.s.L
        L_p = geo.p.L
        L_y = geo.L_y
        L_z = geo.L_z
        tab_n_y = geo.n.centre_y_tab
        tab_n_z = geo.n.centre_z_tab
        L_tab_n = geo.n.L_tab
        tab_p_y = geo.p.centre_y_tab
        tab_p_z = geo.p.centre_z_tab
        L_tab_p = geo.p.L_tab

        geometry = pybamm.battery_geometry(options={"dimensionality": 2})

        assert set([x.name for x in geometry.parameters]) == \
            set(
                [
                    x.name \
                    for x in [
                        L_n,
                        L_s,
                        L_p,
                        L_y,
                        L_z,
                        tab_n_y,
                        tab_n_z,
                        L_tab_n,
                        tab_p_y,
                        tab_p_z,
                        L_tab_p,
                    ] \
                ] \
            )
        assert all(isinstance(x, pybamm.Parameter) for x in geometry.parameters)


