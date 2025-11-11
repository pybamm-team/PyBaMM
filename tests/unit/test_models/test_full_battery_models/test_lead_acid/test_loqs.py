import pytest

import pybamm


class TestLeadAcidLOQS:
    @pytest.mark.parametrize(
        "options",
        [
            {"thermal": "isothermal"},
            {"convection": "uniform transverse"},
            {"dimensionality": 1, "convection": "full transverse"},
        ],
        ids=["isothermal", "with_convection", "with_convection_1plus1d"],
    )
    def test_well_posed(self, options):
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

        if "thermal" in options:
            model = pybamm.lead_acid.LOQS(build=False)
            model.build_model()
            model.check_well_posedness()

    def test_default_geometry(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lead_acid.LOQS(options)
        assert "negative particle" not in model.default_geometry
        assert isinstance(model.default_spatial_methods, dict)
        assert isinstance(
            model.default_spatial_methods["current collector"],
            pybamm.ZeroDimensionalSpatialMethod,
        )
        assert issubclass(
            model.default_submesh_types["current collector"],
            pybamm.SubMesh0D,
        )

    @pytest.mark.parametrize(
        "dimensionality, spatial_method, submesh_type",
        [
            (1, pybamm.FiniteVolume, pybamm.Uniform1DSubMesh),
            (2, pybamm.ScikitFiniteElement, pybamm.ScikitUniform2DSubMesh),
        ],
        ids=["1plus1_d", "2plus1_d"],
    )
    def test_well_posed_differential(
        self, dimensionality, spatial_method, submesh_type
    ):
        options = {
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": dimensionality,
        }
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

        assert isinstance(
            model.default_spatial_methods["current collector"], spatial_method
        )
        assert issubclass(
            model.default_submesh_types["current collector"], submesh_type
        )


class TestLeadAcidLOQSWithSideReactions:
    @pytest.mark.parametrize(
        "options",
        [
            {"surface form": "differential", "hydrolysis": "true"},
            {"surface form": "algebraic", "hydrolysis": "true"},
        ],
        ids=["differential", "algebraic"],
    )
    def test_well_posed(self, options):
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()


class TestLeadAcidLOQSSurfaceForm:
    @pytest.mark.parametrize(
        "options",
        [
            {"surface form": "differential"},
            {"surface form": "algebraic"},
            {
                "surface form": "differential",
                "current collector": "potential pair",
                "dimensionality": 1,
            },
        ],
        ids=["differential", "algebraic", "1plus1_d"],
    )
    def test_well_posed(self, options):
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_default_geometry(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        assert "current collector" in model.default_geometry
        options.update({"current collector": "potential pair", "dimensionality": 1})
        model = pybamm.lead_acid.LOQS(options)
        assert "current collector" in model.default_geometry


class TestLeadAcidLOQSExternalCircuits:
    @pytest.mark.parametrize(
        "options",
        [
            {"operating mode": "voltage"},
            {"operating mode": "power"},
            {"calculate discharge energy": "true"},
        ],
        ids=["voltage", "power", "discharge_energy"],
    )
    def test_well_posed(self, options):
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_function(self):
        def external_circuit_function(variables):
            I = variables["Current [A]"]
            V = variables["Voltage [V]"]
            return (
                V
                + I
                - pybamm.FunctionParameter(
                    "Function", {"Time [s]": pybamm.t}, print_name="test_fun"
                )
            )

        options = {"operating mode": external_circuit_function}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()
