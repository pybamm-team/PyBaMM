import pytest
import pybamm


class TestLeadAcidLOQS:
    def test_well_posed(self):
        options = {"thermal": "isothermal"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

        # Test build after init
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

    def test_well_posed_with_convection(self):
        options = {"convection": "uniform transverse"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

        options = {"dimensionality": 1, "convection": "full transverse"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_1plus1_d(self):
        options = {
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 1,
        }
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()
        assert isinstance(
            model.default_spatial_methods["current collector"], pybamm.FiniteVolume
        )
        assert issubclass(
            model.default_submesh_types["current collector"],
            pybamm.Uniform1DSubMesh,
        )

    def test_well_posed_2plus1_d(self):
        options = {
            "surface form": "differential",
            "current collector": "potential pair",
            "dimensionality": 2,
        }
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()
        assert isinstance(
            model.default_spatial_methods["current collector"],
            pybamm.ScikitFiniteElement,
        )
        assert issubclass(
            model.default_submesh_types["current collector"],
            pybamm.ScikitUniform2DSubMesh,
        )


@pytest.fixture(
    params=[
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
def loqs_model(request):
    options = request.param
    model = pybamm.lead_acid.LOQS(options)
    return model


class TestLeadAcidLOQSSurfaceForm:
    def test_well_posed(self, loqs_model):
        loqs_model.check_well_posedness()

    def test_default_geometry(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        assert "current collector" in model.default_geometry
        options.update({"current collector": "potential pair", "dimensionality": 1})
        model = pybamm.lead_acid.LOQS(options)
        assert "current collector" in model.default_geometry


@pytest.fixture(
    params=[
        {"operating mode": "voltage"},
        {"operating mode": "power"},
        {
            "operating mode": lambda variables: variables["Voltage [V]"]
            + variables["Current [A]"]
            - pybamm.FunctionParameter(
                "Function", {"Time [s]": pybamm.t}, print_name="test_fun"
            )
        },
        {"calculate discharge energy": "true"},
    ],
    ids=["voltage", "power", "function", "discharge_energy"],
)
def loqs_external_circuit_model(request):
    options = request.param
    model = pybamm.lead_acid.LOQS(options)
    return model


class TestLeadAcidLOQSExternalCircuits:
    def test_well_posed(self, loqs_external_circuit_model):
        loqs_external_circuit_model.check_well_posedness()
