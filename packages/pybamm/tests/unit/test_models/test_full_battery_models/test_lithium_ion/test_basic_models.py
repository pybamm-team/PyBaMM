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

    def test_dfn_2d_unstructured(self):
        model = pybamm.lithium_ion.BasicDFN2DUnstructured(element_type="quad")
        model.check_well_posedness()

    def test_dfn_3d_unstructured(self):
        model = pybamm.lithium_ion.BasicDFN3DUnstructured()
        model.check_well_posedness()
