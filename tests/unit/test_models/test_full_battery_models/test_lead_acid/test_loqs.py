#
# Tests for the lead-acid LOQS model
#

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


class TestLeadAcidLOQSWithSideReactions:
    def test_well_posed_differential(self):
        options = {"surface form": "differential", "hydrolysis": "true"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_algebraic(self):
        options = {"surface form": "algebraic", "hydrolysis": "true"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()


class TestLeadAcidLOQSSurfaceForm:
    def test_well_posed_differential(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_algebraic(self):
        options = {"surface form": "algebraic"}
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

    def test_default_geometry(self):
        options = {"surface form": "differential"}
        model = pybamm.lead_acid.LOQS(options)
        assert "current collector" in model.default_geometry
        options.update({"current collector": "potential pair", "dimensionality": 1})
        model = pybamm.lead_acid.LOQS(options)
        assert "current collector" in model.default_geometry


class TestLeadAcidLOQSExternalCircuits:
    def test_well_posed_voltage(self):
        options = {"operating mode": "voltage"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()

    def test_well_posed_power(self):
        options = {"operating mode": "power"}
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

    def test_well_posed_discharge_energy(self):
        options = {"calculate discharge energy": "true"}
        model = pybamm.lead_acid.LOQS(options)
        model.check_well_posedness()
