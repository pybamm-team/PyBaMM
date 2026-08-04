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

    @pytest.mark.filterwarnings(
        "ignore:Could not determine how to combine submeshes"
    )
    def test_dfn_2d_vector_field_variable(self):
        # VectorField variables on structured 2D meshes must stay on the
        # scalar processing path (per-component casadi lists are only for
        # unstructured meshes)
        model = pybamm.lithium_ion.BasicDFN2D()
        var_pts = {k: 5 for k in model.default_var_pts}
        sim = pybamm.Simulation(model, var_pts=var_pts)
        solution = sim.solve([0, 10])
        current_density = solution["Electrolyte current density [A.m-2]"]
        assert np.all(np.isfinite(current_density.data))
        assert np.all(np.isfinite(current_density(t=5)))
