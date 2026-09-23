#
# Tests for the basic lithium-ion models
#
import numpy as np
import pytest

import pybamm


class TestBasicModels:
    def test_dfn_well_posed(self):
        model = pybamm.lithium_ion.BasicDFN()
        model.check_well_posedness()

    def test_spm_well_posed(self):
        model = pybamm.lithium_ion.BasicSPM()
        model.check_well_posedness()

    def test_dfn_half_cell_well_posed(self):
        options = {"working electrode": "positive"}
        model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)
        model.check_well_posedness()

    def test_dfn_half_cell_total_lithium_in_electrolyte(self):
        model = pybamm.lithium_ion.BasicDFNHalfCell(
            options={"working electrode": "positive"}
        )
        parameter_values = model.default_parameter_values
        solution = pybamm.Simulation(model, parameter_values=parameter_values).solve(
            [0, 1]
        )

        area = (
            parameter_values["Electrode width [m]"]
            * parameter_values["Electrode height [m]"]
            * parameter_values[
                "Number of electrodes connected in parallel to make a cell"
            ]
        )
        expected_electrolyte_lithium = (
            area
            * parameter_values["Initial concentration in electrolyte [mol.m-3]"]
            * (
                parameter_values["Separator porosity"]
                * parameter_values["Separator thickness [m]"]
                + parameter_values["Positive electrode porosity"]
                * parameter_values["Positive electrode thickness [m]"]
            )
        )

        np.testing.assert_allclose(
            solution["Total lithium in electrolyte [mol]"](0),
            expected_electrolyte_lithium,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_dfn_composite_well_posed(self):
        model = pybamm.lithium_ion.BasicDFNComposite()
        model.check_well_posedness()

    def test_dfn_2d(self):
        model = pybamm.lithium_ion.BasicDFN2D()
        model.check_well_posedness()

    @pytest.mark.filterwarnings("ignore:Could not determine how to combine submeshes")
    def test_dfn_2d_vector_field_variable(self):
        # A VectorField variable on a structured 2D mesh cannot be read
        # directly, but must fail with guidance rather than an opaque error,
        # and extracting a component must work.
        model = pybamm.lithium_ion.BasicDFN2D()
        model.variables["Electrolyte current density x [A.m-2]"] = pybamm.Component(
            model.variables["Electrolyte current density [A.m-2]"], 0
        )
        var_pts = {k: 5 for k in model.default_var_pts}
        sim = pybamm.Simulation(model, var_pts=var_pts)
        solution = sim.solve([0, 10])

        with pytest.raises(NotImplementedError, match=r"pybamm\.Component"):
            solution["Electrolyte current density [A.m-2]"]

        component = solution["Electrolyte current density x [A.m-2]"]
        assert np.all(np.isfinite(component(t=5)))

    @pytest.mark.parametrize(
        ("dimensionality", "element_type"),
        [(1, "quad"), (1, "triangle"), (2, "hexahedron"), (2, "tetrahedron")],
    )
    def test_dfn_unstructured(self, dimensionality, element_type):
        model = pybamm.lithium_ion.BasicDFNUnstructured(
            {"dimensionality": dimensionality}
        )
        model.check_well_posedness()

        submesh_types = model.default_submesh_types
        for domain in ["negative electrode", "separator", "positive electrode"]:
            submesh_types[domain] = pybamm.UnstructuredMeshGenerator(
                element_type=element_type
            )
        sim = pybamm.Simulation(
            model,
            submesh_types=submesh_types,
            var_pts={k: 3 for k in model.default_var_pts},
        )
        sim.build()
        for domain in ["negative electrode", "separator", "positive electrode"]:
            assert sim.mesh[domain].dimension == dimensionality + 1
            assert sim.mesh[domain].element_type == element_type

    @pytest.mark.parametrize(
        ("options", "element_type"),
        [(None, "quad"), ({"dimensionality": 2}, "hexahedron")],
    )
    def test_dfn_unstructured_default_element_type(self, options, element_type):
        model = pybamm.lithium_ion.BasicDFNUnstructured(options)
        sim = pybamm.Simulation(model, var_pts={k: 3 for k in model.default_var_pts})
        sim.build()
        assert sim.mesh["negative electrode"].element_type == element_type

    def test_dfn_unstructured_default_dimensionality(self):
        model = pybamm.lithium_ion.BasicDFNUnstructured()
        assert model.options["dimensionality"] == 1

    @pytest.mark.parametrize("dimensionality", [0, 3])
    def test_dfn_unstructured_bad_dimensionality(self, dimensionality):
        with pytest.raises(pybamm.OptionError, match=r"1 \(x-z mesh\) or 2"):
            pybamm.lithium_ion.BasicDFNUnstructured(
                {"dimensionality": dimensionality, "cell geometry": "pouch"}
            )
